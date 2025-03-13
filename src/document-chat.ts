import { readFile, writeFile } from 'node:fs/promises'
import { createReadStream, readFileSync } from 'node:fs'
import path from 'node:path'
import pdfParser from 'pdf-parse'
import ollama from 'ollama'
import { QdrantClient } from '@qdrant/js-client-rest'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatGroq } from '@langchain/groq'

/* 
  set this to true to create a new collection
  in the vector db and populate it with vector data
  from the book
*/
const initializeCollection = false
const GROQ_API_KEY = process.env.GROQ_API_KEY

const qdrant = new QdrantClient({
  host: 'localhost',
  port: 6333
})

if (initializeCollection) {
  const __dirname = import.meta.dirname

  const pdfFile = await readFile(
    path.join(__dirname, 'files', 'o-pequeno-principe.pdf')
  )

  const { text } = await pdfParser(pdfFile)

  await writeFile(
    path.join(__dirname, 'files', 'o-pequeno-principe.txt'),
    text.trim()
  )

  const textFile = createReadStream(
    path.join(__dirname, 'files', 'o-pequeno-principe.txt')
  )

  const chunkSize = 1000
  let currentChunk = ''
  let id = 1

  const collectionAlreadyExists = await qdrant.collectionExists('pdf-documents')

  if (!collectionAlreadyExists.exists) {
    qdrant.createCollection('pdf-documents', {
      vectors: {
        size: 1024,
        distance: 'Cosine'
      }
    })
  }

  textFile
    .on('data', async chunk => {
      currentChunk += chunk

      while (currentChunk.length >= chunkSize) {
        const pdfChunk = currentChunk.slice(0, chunkSize)

        const { embedding } = await ollama.embeddings({
          model: 'mxbai-embed-large',
          prompt: pdfChunk
        })

        qdrant.upsert('pdf-documents', {
          points: [
            {
              id,
              vector: embedding,
              payload: {
                content: pdfChunk
              }
            }
          ]
        })

        currentChunk = currentChunk.slice(chunkSize)

        id++
      }
    })
    .on('end', async () => {
      if (currentChunk.length > 0) {
        const { embedding } = await ollama.embeddings({
          model: 'mxbai-embed-large',
          prompt: currentChunk
        })

        qdrant.upsert('pdf-documents', {
          points: [
            {
              id,
              vector: embedding,
              payload: {
                content: currentChunk
              }
            }
          ]
        })
      }
    })
}

const llm = new ChatGroq({
  model: 'mixtral-8x7b-32768',
  temperature: 0,
  maxRetries: 2,
  streaming: true,
  maxTokens: undefined,
  apiKey: GROQ_API_KEY
})

const question = 'o que é essencial?'

const { embedding } = await ollama.embeddings({
  model: 'mxbai-embed-large',
  prompt: `${question}`
})

const { points } = await qdrant.query('pdf-documents', {
  query: embedding,
  with_payload: true,
  limit: 3
})

let content = ''
points.forEach(point => (content += point.payload?.content))

console.log(content)

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
  {
    role: 'user',
    content: `${question}`
  }
])

const chain = prompt.pipe(llm).pipe(new StringOutputParser())

const response = chain.streamEvents(
  {
    file: content
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
