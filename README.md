# MDD-5k
Official Implementation of AAAI 2025 paper "MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents"

## Dataset Overview
MDD-5k is build upon 926 unique patient files, and each patient file is used to synthesize five diagnostic conversations, resulting in a total of 4630 conversations. 

The synthetic conversations and corresponding lables can be found in `./MDD_5k` and `./Label`. Each json file in `./MDD_5k` contains five conversations and all the five conversation is generated based on the same patient file.


## Preprocessing Steps
Run `python patient_template_gen.py` to get statistics of MDD-5k dataset and generate fictitious patient experiences

## Synthesize Diagnostic Conversation
Current code only supports using deployed model for generation. You can either access state-of-the-art GPT model through OpenAI API keys, or deploying local models with vLLM. For the former way, enter your OpenAI key in line 37 of `llm_tools_api.py`. For the latter way, enter the key and server host in line 44 and 45 of `llm_tools_api.py`. For OpenAI LLMs, we found the performance of `gpt-4o` is the best.

Then run `python main.py` to start synthesizing data.
## Example
One real patient case is shown in `./raw_data/pa20.json`. Five fictitious patient experiences generated with this patient case are shown in `./prompts/background_story/patient_1`. The synthesized diagnostic conversations are in `./DataSyn/patient1.json`. The complete MDD-5k dataset will release when the ethics review finishes.
