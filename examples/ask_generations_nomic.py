# This file uses gpt4all-lora-quantized.bin model
# it's based on Alpaca-LoRA  https://github.com/tloen/alpaca-lora
# model should automatically download.

# needs 'nomic' as a pip dependency to be installed
# for more info on how to LoRA works: https://arxiv.org/pdf/2106.09685.pdf

from pygpt4all import GPT4All_J

#m = GPT4All("./ggml-gpt4all-l13b-snoozy.bin")
model = GPT4All_J("ggml-gpt4all-j-v1.3-groovy.bin")

while True:
    try:
        prompt = input("You: ", flush=True)
        if prompt == '':
            continue
        elif prompt == "stop":
            break
        print(f"AI:", end='')
        for token in model.generate(prompt):
            print(f"{token}", end='', flush=True)
        print()
    except KeyboardInterrupt:
        break


# available parameters in model.generate:
'''
x.generate(prompt,
    max_tokens=200,
    temp=0.7,
    top_k=40,
    top_p=0.4, repeat_penalty=1.18,
    repeat_last_n=64,
    n_batch=8,
    n_predict=None,
    streaming=False,
    callback=pyllmodel.empty_response_callback
) 
'''

# link to the documentation: https://docs.gpt4all.io/gpt4all_python.html#gpt4all.gpt4all.GPT4All.generate
