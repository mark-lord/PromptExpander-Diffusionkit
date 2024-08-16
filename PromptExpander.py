import subprocess
import re
import threading
import queue
import os
import signal
import math
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import fire
import mlx_lm

# Constants
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_SEED = 0
DEFAULT_MODEL = "mlx-community/Gemma-2-9B-It-SPPO-Iter3-4bit"
DEFAULT_MAX_TOKENS = 150
DEFAULT_TOP_P = 0.1
DEFAULT_TEMPERATURE = 0.3

def sanitize_filename(filename):
    """Remove any characters from the filename that aren't alphanumeric, underscore, hyphen, period, or space."""
    return re.sub(r'[^\w\-_\. ]', '_', filename)

def adjust_dimensions(width, height):
    """Adjust dimensions to be multiples of 64."""
    return math.ceil(width / 64) * 64, math.ceil(height / 64) * 64

def generate_image_prompt(prompt, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):

    """Generate an expanded image prompt using MLX-LM."""
    model_obj, tokenizer = mlx_lm.load(model)

    full_prompt = f"""<bos><system>
You are a prompt expander. You expand out a request for an image prompt with details and such like. 
You have granular control over the image aspect ratio. Your range is from 400px -> 1024px for width (X) and height (Y). Use the flag ~(/X)x(/Y) to specify. 
If the user's initial unexpanded image prompt is of a landscape or cinematic shot, make a wide aspect ratio.
If the user's initial unexpanded image prompt is of a single character or entity, make a tall aspect ratio.
If they request an image with two or more characters in it, opt for a wider aspect ratio.
Square aspect ratio is good for icons or item close-ups and such. 
<start_of_turn>user
Hi! I need you to expand out a request for an image prompt with details and such like. 
For instance, if I ask for 'a photo of someone holding a cellphone', I'd like you to expand it out. 
Does that seem like something you can do?
Don't bother writing anything after you've created your expanded prompt.<end_of_turn>
<start_of_turn>model
Of course! Will generate that for you. Here's your expanded prompt:
A close-up photo of a person holding a smartphone in their hand. 
The person's hand is visible, with the phone being held upright. 
The background is slightly blurred, emphasizing the phone. 
The phone screen is on, displaying a generic app or home screen. 
The hand and phone are the primary focus, with natural lighting casting soft shadows. 
The person is casually dressed, with no identifiable logos or text. ~512x768<end_of_turn>
<start_of_turn>user
Fantastic! Now could you do it for this prompt?
'{prompt}'<end_of_turn>
<start_of_turn>model
Ooh, great image idea! Very happy to expand that for you. Here's the expanded prompt:"""

    messages = [{"role": "user", "content": full_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = mlx_lm.generate(
        model_obj,
        tokenizer,
        formatted_prompt,
        max_tokens,
        temp=temperature,
        verbose=False,
    )

    # Extract the expanded prompt from the response
    expanded_prompt = response.split("Here's the expanded prompt:")[-1].strip()
    expanded_prompt = expanded_prompt.split("<end_of_turn>")[0].strip()
    return expanded_prompt

def run_diffusionkit(prompt, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, seed=DEFAULT_SEED, save=False):
    """Run the diffusionkit-cli command to generate an image."""
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_prompt = sanitize_filename(prompt)[:50]
        output_path = f"{sanitized_prompt}_{timestamp}_seed{seed}.png"
    else:
        output_path = "flux_output.png"

    # Remove any remaining ~ flags from the prompt
    clean_prompt = re.sub(r'~\w+', '', prompt).strip()

    command = [
        "diffusionkit-cli",
        "--prompt", clean_prompt,
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--output-path", output_path,
        "--model-version", "FLUX.1-schnell",
        "--steps", "4"
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    
    return process.returncode == 0, output_path, process.stderr

def parse_input(input_string):
    """Parse user input for special flags and extract prompt."""
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    save = False
    lm_off = False
    
    # Extract dimensions if provided
    dim_match = re.search(r'~(\d+)x(\d+)', input_string)
    if dim_match:
        width, height = map(int, dim_match.groups())
        input_string = re.sub(r'~\d+x\d+', '', input_string)
    
    # Check for save flag
    if '~save' in input_string:
        save = True
        input_string = input_string.replace('~save', '')
    
    # Check for lm-off flag
    if '~lm-off' in input_string:
        lm_off = True
        input_string = input_string.replace('~lm-off', '')
    
    prompt = input_string.strip()
    return prompt, width, height, save, lm_off

def truncate_prompt(prompt, target_length=350):
    """
    Truncate the prompt to the target length, then remove words if needed.
    
    :param prompt: The prompt to truncate
    :param target_length: The initial target length to truncate to (default 320)
    :return: Truncated prompt
    """
    # First, truncate to target length
    if len(prompt) > target_length:
        prompt = prompt[:target_length].rsplit(' ', 1)[0].strip()
    
    # If we need to truncate further, remove words one by one
    words = prompt.split()
    while len(prompt) > target_length and len(words) > 1:
        words = words[:-1]
        prompt = ' '.join(words)
    
    return prompt

def worker(task_queue):
    """Worker function to process tasks from the queue."""
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        prompt, width, height, seed, save = task["prompt"], task["width"], task["height"], task["seed"], task["save"]
        original_prompt = prompt
        original_width, original_height = width, height
        
        # Extract size flag if present
        size_flag_match = re.search(r'(~\d+x\d+)', prompt)
        if size_flag_match:
            size_flag = size_flag_match.group(1)
            content = prompt.replace(size_flag, '').strip()
            width, height = map(int, size_flag.replace('~', '').split('x'))
        else:
            size_flag = f"~{width}x{height}"
            content = prompt

        success = False
        dimensions_adjusted = False
        target_length = 320
        
        while not success:
            content = truncate_prompt(content, target_length)
            full_prompt = f"{content} {size_flag}".strip()
            print(f"Attempting to generate image with prompt length: {len(full_prompt)} characters")
            success, output_path, error = run_diffusionkit(full_prompt, width, height, seed, save)
            if success:
                print(f"Image generated successfully: {output_path}")
                print(f"Final prompt used: {content}")  # Show the final truncated prompt
                if full_prompt != original_prompt:
                    print(f"Note: Prompt was truncated from {len(original_prompt)} to {len(full_prompt)} characters")
                if width != original_width or height != original_height:
                    print(f"Note: Dimensions were adjusted from {original_width}x{original_height} to {width}x{height}")
            else:
                if not dimensions_adjusted:
                    print("Image generation failed. Adjusting dimensions...")
                    width, height = adjust_dimensions(width, height)
                    size_flag = f"~{width}x{height}"
                    dimensions_adjusted = True
                else:
                    print("Image generation failed. Truncating prompt further...")
                    target_length -= 10  # Reduce target length by 10 characters each iteration
                    if target_length <= 0:
                        print("Error: Unable to generate image with current prompt. Skipping this task.")
                        break
        
        task_queue.task_done()
    """Truncate the prompt by removing the last word."""
    words = prompt.split()
    if len(words) > 1:
        return " ".join(words[:-1])
    return ""

def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    raise KeyboardInterrupt

def main():
    """Main function to run the image generation script."""
    signal.signal(signal.SIGINT, signal_handler)

    last_prompt = ""
    last_seed = DEFAULT_SEED
    task_queue = queue.Queue()

    worker_thread = threading.Thread(target=worker, args=(task_queue,))
    worker_thread.start()

    session = PromptSession(history=InMemoryHistory())

    print("Image Generator Activated")
    print("\nINSTRUCTIONS:")
    print("Use ~WIDTHxHEIGHT to specify dimensions (e.g., ~1024x768)")
    print("Use ~save to save with a unique filename")
    print("Use ~lm-off to skip LLM prompt expansion")
    print("Use up/down arrows to navigate through prompt history")
    print("You can queue multiple prompts")
    print("Leaving the prompt blank will re-use the most recent prompt with seed+1")
    print("Press Ctrl+C to exit")
    print("\nEnter your prompt:\n")
    
    try:
        while True:
            try:
                user_input = session.prompt("> ")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            prompt, width, height, save, lm_off = parse_input(user_input)

            if not prompt:
                prompt = last_prompt
                last_seed += 1
            else:
                last_prompt = prompt
                last_seed = DEFAULT_SEED

            if not lm_off:
                print("Generating expanded prompt...")
                expanded_prompt = generate_image_prompt(prompt)
                print(f"Expanded prompt: {expanded_prompt}")
                prompt = expanded_prompt

            print(f"Queueing image generation...")
            task_queue.put({"prompt": prompt, "width": width, "height": height, "seed": last_seed, "save": save})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        task_queue.put(None)
        worker_thread.join()
        print("Exiting the program. Goodbye!")

if __name__ == "__main__":
    fire.Fire(main)