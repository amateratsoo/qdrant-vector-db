import { readFileSync } from 'node:fs'
import path from 'node:path'

import { ChatGroq } from '@langchain/groq'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'

import { input } from '@inquirer/prompts'

const GROQ_API_KEY = process.env.GROQ_API_KEY

const __dirname = import.meta.dirname

const file = readFileSync(path.join(__dirname, 'files', 'file.txt'), {
  encoding: 'utf-8'
})

const llm = new ChatGroq({
  model: 'mixtral-8x7b-32768',
  temperature: 0,
  maxRetries: 2,
  streaming: true,
  maxTokens: undefined,
  apiKey: GROQ_API_KEY
})

async function askPrompt() {
  const question = await input({
    message: '🤖 Pergunte qualquer coisa acerca do documento:',
    required: true
  })

  const prompt = ChatPromptTemplate.fromMessages([
    {
      role: 'system',
      content: `
        Você é uma IA projetada para responder exclusivamente com base nas informações contidas no documento que eu fornecerei. Nenhuma informação externa pode ser utilizada, e você não pode fazer suposições nem inventar respostas. Se a resposta não estiver explícita no documento, você deve dizer: "Desculpe, essa informação não está disponível no documento" e nada mais.

        Você deve manter uma conversa natural, sendo educado e fluente, mas sem se desviar do conteúdo. Se eu perguntar algo que não está no documento, você não deve buscar respostas externas. Sempre que possível, forneça respostas claras e objetivas, diretamente relacionadas ao conteúdo do documento.

        **Importante:**
        - Não adicione nenhuma informação que não esteja no documento.
        - Caso a informação que eu pedir não esteja no documento, seja claro e direto: "Desculpe, essa informação não está disponível no documento."
        - O seu objetivo é fornecer respostas úteis dentro dos limites do que está no conteúdo do documento. Não se preocupe em fornecer mais contexto ou explicar o que não está no texto.
        - Não faça conjecturas ou suposições.

        Agora, você está pronto para começar a conversa. Fique à vontade para me perguntar qualquer coisa relacionada ao conteúdo que está no documento.

        O documento é este: {file}
      `
    },
    { role: 'user', content: question }
  ])

  const chain = prompt.pipe(llm).pipe(new StringOutputParser())

  const response = chain.streamEvents(
    {
      file
    },
    {
      version: 'v2'
    }
  )

  for await (const stream of response) {
    if (stream.event === 'on_chat_model_stream') {
      process.stdout.write(stream.data.chunk.content)
    }
  }

  console.log('\n')

  askPrompt()
}

await askPrompt()
