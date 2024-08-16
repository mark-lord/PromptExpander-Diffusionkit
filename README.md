# PromptExpander-Diffusionkit
A little file for doing LLM-assisted prompt expansion and image generation using Flux.schnell - complete with prompt history, prompt queueing, LLM prompt expansion and aspect ratio picker - all on the command line, all in MLX :)

## Instructions to get Diffusionkit on your Mac:
1. Install Miniconda for macOS
2. Run this in terminal:

conda create -n diffusionkit python=3.11 -y
conda activate diffusionkit
pip install diffusionkit

3. Download flux.schnell and test it's working using this in terminal:

diffusionkit-cli --prompt "A robot holding a sign that says 'FLUX RULES!'" --output-path image.png --model-version FLUX.1-schnell --steps 4

## Instructions to run PromptExpander:
1. Download the PromptExpander.py file
2. Run this in terminal:

Pip3 install fire mlx_lm prompt_toolkit

4. Then run this to start up the PromptExpander CLI tool:

python PromptExpander.py

4. Go through first time set up as it downloads Gemma-9b-SPPO
5. Type in your prompts, instructions are given to you on the command-line
6. Enjoy!
7. (Optional): Go into PromptExpander.py and edit the full_prompt to your heart's content; tinkering with the single shot example provided by the model, as well as the system card, will affect the style of future images
