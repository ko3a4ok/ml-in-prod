from openai import OpenAI
import json

client = OpenAI()

data = []
for _ in range(3):
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role": "system",
              "content": 'generate a list("data") of 2 elements each with (DDL table schema, natural language request, and corresponding SQL query). return as json array with "schema", "request" and "query" for each element'
          },
      ],
      temperature=1,
      max_tokens=512,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
  )
  data.extend(json.loads(response.choices[0].message.content)['data'])
with open('data_raw.json', 'w') as f:
  f.write(json.dumps(data, indent=2))
