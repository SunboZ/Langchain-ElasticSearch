from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("/home/ubuntu/opt/text-generation-webui/models", model_file="yi-34b-chat.Q5_K_S.gguf", model_type="yi", gpu_layers=50)

print(llm("AI is going to"))

