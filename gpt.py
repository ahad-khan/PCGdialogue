import openai

items = []
items_file = open('output.txt', 'r').read()
data = items_file.split("\n")
data = list(filter(None, data))
print(data)

openai.api_key = open('key.txt', 'r').read()
message_history = []

def call(user_input, role="user"):
    print("User input >> ", user_input)
    message_history.append({"role": role, "content": user_input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    reply_content = completion.choices[0].message.content
    with open("item_description.txt", "a") as f:
        print(reply_content, '\n\n' + '_' * 80, file=f)
    message_history.append({"role": "assistant", "content": reply_content})
    return reply_content


for i in range(1):
    user_input = "Write a brief description of {0} as if it were an item in a fantasy game.".format(data[38])
    print(user_input)
    print()
    call(user_input)
    print()

print(message_history)
