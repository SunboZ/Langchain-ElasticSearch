from llama_cpp import Llama


llm = Llama(
      model_path="/home/ubuntu/opt/text-generation-webui/models/yi-34b-chat.Q5_K_S.gguf",
      chat_format="llama-2",
      n_gpu_layers=-1,
      n_ctx=1024
)

print("llm loaded")
history = []
_ = input("-")
while True:
    question = input("Question: ")

    message = {"role": "user", "content": question}
    history.append(message)
    res = llm.create_chat_completion(
        messages = history
    )
    print(res)
    response = res.get("choices")[-1].get("message").get("content").split("<|im_end|>")[0].strip()
    history += [{"role": "assistant", "content": response}]
    