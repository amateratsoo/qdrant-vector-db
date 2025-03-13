import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import pdfParser from 'pdf-parse'
import ollama from 'ollama'
import { QdrantClient } from '@qdrant/js-client-rest'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatGroq } from '@langchain/groq'

interface Point {
  id: string | number
  version: number
  score: number
  payload?:
    | Record<string, unknown>
    | {
        [key: string]: unknown
      }
    | null
    | undefined
  vector?:
    | Record<string, unknown>
    | number[]
    | number[][]
    | {
        [key: string]:
          | number[]
          | number[][]
          | {
              text: string
              model?: string | null | undefined
            }
          | {
              indices: number[]
              values: number[]
            }
          | undefined
      }
    | {
        text: string
        model?: string | null | undefined
      }
    | null
    | undefined
  shard_key?: string | number | Record<string, unknown> | null | undefined
  order_value?: number | Record<string, unknown> | null | undefined
}

/* 
  set this to true to create a new collection
  in the vector db and populate it with vector data
  from the book
*/
const initializeCollection = true
const documentName = 'resumo-de-cg'

const GROQ_API_KEY = process.env.GROQ_API_KEY

const qdrant = new QdrantClient({
  host: 'localhost',
  port: 6333
})

/**
 * Function to split text into chunks
 */
const chunkText = (text: string, maxLength = 1000) => {
  const sentences = text.split(/(?<=[.?!])\s+/) // split by sentences
  const chunks = []
  let chunk = ''

  sentences.forEach(sentence => {
    if ((chunk + sentence).length > maxLength) {
      chunks.push(chunk.trim())
      chunk = ''
    }
    chunk += sentence + ' '
  })

  if (chunk.trim()) {
    chunks.push(chunk.trim())
  }

  return chunks
}

/**
 * Function to merge chunks into a single coherent context
 */
function mergeChunks(points: Point[]) {
  const uniqueChunks = new Set() // prevent duplications

  points.forEach(point => {
    const content = point.payload?.content || ''
    uniqueChunks.add((content as string).trim())
  })

  return Array.from(uniqueChunks).join('\n\n')
}
async function main() {
  if (initializeCollection) {
    const __dirname = import.meta.dirname

    const pdfFile = await readFile(
      path.join(__dirname, 'files', `${documentName}.pdf`)
    )

    const { text } = await pdfParser(pdfFile)

    await writeFile(
      path.join(__dirname, 'files', `${documentName}.txt`),
      text.trim()
    )

    const textChunks = chunkText(text.trim(), 1000)

    const collectionAlreadyExists = await qdrant.collectionExists(documentName)

    if (collectionAlreadyExists.exists) {
      console.log('🔥 Deleting collection with the same name...\n')
      const collectionDeleted = await qdrant.deleteCollection(documentName)

      if (!collectionDeleted) {
        console.log(
          `📍 Something went wrong, we couldn't delete the collection ${documentName}\n`
        )
        return
      }

      console.log(`🎉 Collection ${documentName} deleted succesfully\n`)
    }

    const collectionCreadted = await qdrant.createCollection(documentName, {
      vectors: {
        size: 1024,
        distance: 'Cosine'
      }
    })

    if (!collectionCreadted) {
      console.log(
        `📍 Something went wrong, we couldn't create the collection ${documentName}\n`
      )
      return
    }

    console.log(`🎉 Collection ${documentName} created succesfully\n`)

    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i]

      const { embedding } = await ollama.embeddings({
        model: 'mxbai-embed-large',
        prompt: chunk
      })

      await qdrant.upsert(documentName, {
        points: [
          {
            id: i + 1,
            vector: embedding,
            payload: {
              content: chunk
            }
          }
        ]
      })
    }

    console.log(`✨ Collection ${documentName} is set and ready`)
    return
  }

  const llm = new ChatGroq({
    model: 'mixtral-8x7b-32768',
    temperature: 1,
    maxRetries: undefined,
    streaming: true,
    maxTokens: undefined,
    apiKey: GROQ_API_KEY
  })

  const question = 'Qual é a lição que o pequeno príncipe aprende com a raposa?'

  // generate the question as an embedding
  const { embedding } = await ollama.embeddings({
    model: 'mxbai-embed-large',
    prompt: `${question}`
  })

  // get relevant points
  const { points } = await qdrant.query(documentName, {
    query: embedding,
    with_payload: true,
    limit: 6
  })

  const content = mergeChunks(points)

  // console.log('Contexto retornado do Qdrant:\n', content)

  const prompt = ChatPromptTemplate.fromMessages([
    {
      role: 'system',
      content: `
          Você é uma IA projetada para responder e conversar exclusivamente com base nas informações contidas no documento que eu fornecerei. Nenhuma informação externa pode ser utilizada, e você não pode fazer suposições nem inventar respostas. Se a resposta não estiver explícita no documento, você deve dizer: "Desculpe, essa informação não está disponível no documento" e nada mais.

          Você deve manter uma conversa natural, sendo educado e fluente, mas sem se desviar do conteúdo. Se eu perguntar algo que não está no documento, você não deve buscar respostas externas. Sempre que possível, forneça respostas claras e objetivas, diretamente relacionadas ao conteúdo do documento.

          **Importante:**
          - Não adicione nenhuma informação que não esteja no documento.
          - Caso a informação que eu pedir não esteja no documento, seja claro e direto: "Desculpe, essa informação não está disponível no documento."
          - O seu objetivo é fornecer respostas úteis dentro dos limites do que está no conteúdo do documento. Não se preocupe em fornecer mais contexto ou explicar o que não está no texto.
          - Não faça conjecturas ou suposições, mas se necessário, dê um contexto que esteja no documento e que ajude a responder a pergunta mais claramente.
          - Seja amigável e prestável.
          - Dê respostas completas que façam sentido a pergunta.

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

  let chatResponse = ''

  for await (const stream of response) {
    if (stream.event === 'on_chat_model_stream') {
      process.stdout.write(stream.data.chunk.content)

      chatResponse += stream.data.chunk.content
    }
  }

  console.log('\n')
}

main()
