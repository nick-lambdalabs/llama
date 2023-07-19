from typing import Optional

import fire
import sys

from llama import Llama

def main(
    ckpt_dir: str = "llama-2-7b-chat",
    tokenizer_path: str = "tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096, # can we make this larger?
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    system_message: Optional[str] = None,
):  
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    conversation = []

    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})
        print("System message set to: {}".format(system_message))

    for line in sys.stdin:
        if line.strip() == "exit":
            break

        if line.strip() == "reset":
            conversation = []
            print("Conversation reset")
            continue

        if line.strip().startswith("system:"):
            if len(conversation) != 0:
                print("Reset the conversation first")
                continue
            system_message = line.strip()[len("system:"):].strip()
            conversation.append({"role": "system", "content": system_message})
            print("System message set to: {}".format(system_message))
            continue

        conversation.append({"role": "user", "content": line.strip()})
        
        result = generator.chat_completion(
            [conversation],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        generation = result[0]["generation"]
        conversation.append(generation)
        print("=====================================")
        print(generation["content"])
        print("=====================================")


if __name__ == "__main__":
    fire.Fire(main)
