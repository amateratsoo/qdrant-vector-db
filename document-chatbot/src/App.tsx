import { Menu, BirdIcon, Paperclip, Brain, X } from 'lucide-react'
import { useState } from 'react'
import ollama from 'ollama'
import { QdrantClient } from '@qdrant/js-client-rest'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatGroq } from '@langchain/groq'

const documents = [
  { title: 'O Pequeno Príncipe', collectionName: 'o-pequeno-principe' },
  { title: 'Resumo de CG', collectionName: 'resumo-de-cg' }
]

const qdrant = new QdrantClient({
  host: 'localhost',
  port: 6333
})

const llm = new ChatGroq({
  model: 'mixtral-8x7b-32768',
  temperature: 0.7,
  maxRetries: undefined,
  streaming: true,
  maxTokens: undefined,
  apiKey: process.env.GROQ_API_KEY
})

interface Chats {
  from: 'human' | 'ai'
  message: string
}

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

export function App() {
  const [currentDocument, setCurrentDocument] = useState('o-pequeno-principe')
  const [chats, setChats] = useState<Chats[]>([])
  const [context, setContext] = useState([])
  const [loading, setLoading] = useState(false)
  const [showSidebar, setShowSidebar] = useState(false)

  const { title, collectionName } = documents.filter(
    ({ collectionName }) => collectionName === currentDocument
  )[0]

  /**
   * Function to merge chunks into a single coherent context
   */
  function mergeChunks(points: Point[]) {
    const uniqueChunks = new Set() // prevent duplications

    points.forEach(point => {
      const content = point.payload?.content || ''
      uniqueChunks.add((content as string).trim())
    })

    setContext(Array.from(uniqueChunks) as [])
    return Array.from(uniqueChunks).join('\n\n')
  }

  async function startPrompt({ message }: { message: string }) {
    setChats(prev => [...prev, { from: 'human', message }])

    setLoading(true)

    // generate the question as an embedding
    const { embedding } = await ollama.embeddings({
      model: 'mxbai-embed-large',
      prompt: `${message}`
    })

    // get relevant points
    const { points } = await qdrant.query(collectionName, {
      query: embedding,
      with_payload: true,
      limit: 6
    })

    const content = mergeChunks(points)

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
        content: `${message}`
      }
    ])

    const chain = prompt.pipe(llm).pipe(new StringOutputParser())

    const response = await chain.invoke({
      file: content
    })

    setLoading(false)
    setChats(prev => [...prev, { from: 'ai', message: response }])
  }

  return (
    <div className='min-h-screen bg-white flex flex-col overflow-hidden'>
      <div
        className={`fixed h-screen w-screen top-0 bg-zinc-950/55 z-50 ${
          showSidebar ? 'visible' : 'invisible pointer-events-none'
        }`}
      >
        <button
          className='absolute top-2 left-96'
          onClick={() => setShowSidebar(false)}
        >
          <X className='size-6 text-zinc-300' />
        </button>
        <aside
          className={`h-screen w-80 transition-transform delay-75 ${
            showSidebar ? '-translate-x-0' : '-translate-x-96'
          } bg-white relative overflow-x-hidden- overflow-y-auto`}
        >
          {/* <button
            className='absolute top-2 left-96'
            onClick={() => setShowSidebar(false)}
          >
            <X className='size-6 text-zinc-300' />
          </button> */}
          <div>
            <h2 className='text-2xl ml-4 pt-4 pb-2 font-serif'>
              <Paperclip className='size-4 inline' /> Documentos
            </h2>

            {documents.map(({ collectionName, title }) => {
              return (
                <button
                  key={collectionName}
                  className={`rounded-md p-1.5 w-[calc(100%-2rem)] mx-auto my-2 block text-start ${
                    currentDocument === collectionName
                      ? 'bg-zinc-800 text-zinc-200'
                      : 'bg-zinc-100'
                  }`}
                  onClick={() => {
                    setCurrentDocument(collectionName)
                    setChats([])
                    setContext([])
                  }}
                >
                  {title}
                </button>
              )
            })}
          </div>

          <div>
            <h2 className='text-2xl ml-4 pt-4 pb-2 font-serif'>
              <Brain className='size-4 inline' /> Contexto
            </h2>

            <span className='rounded-md p-1.5 w-[calc(100%-2rem)] mx-auto my-2 block text-start'>
              {context.length == 0 && 'sem contexto disponível'}
              {context.map((ctx, index) => {
                return (
                  <div
                    className='border-b border-zinc-200 py-4 first:pt-0'
                    key={index}
                  >
                    <span className='rounded-md bg-zinc-200 text-zinc-400 px-2 mr-2 text-base inline-block'>
                      {index + 1}
                    </span>
                    {ctx}
                  </div>
                )
              })}
            </span>
          </div>
        </aside>
      </div>

      <header className='p-6 border-b border-gray-100 fixed w-screen z-20 bg-white'>
        <div className='flex justify-center items-center relative'>
          <button
            className='p-2 hover:bg-gray-50 rounded-lg transition-colors absolute left-5'
            onClick={() => setShowSidebar(true)}
          >
            <Menu className='w-5 h-5 text-gray-600' />
            <span className='sr-only'>Toggle Sidebar</span>
          </button>
          <h1 className='font-medium'>{title}</h1>
        </div>

        {/* <button className='p-2 hover:bg-gray-50 rounded-lg transition-colors flex items-center justify-center gap-2 bg-zinc-100 text-sm'>
            <PlusCircle className='w-5 h-5 text-gray-600' />
            <span className=''>novo chat</span>
          </button> */}
      </header>

      <main className='flex-1 mb-20 flex flex-col items-center p-4 relative z-10 pt-24'>
        <div className='w-full max-w-[40rem] flex flex-1 flex-col justify-end gap-4 overflow-y-auto'>
          {chats.length === 0 ? (
            <div className='flex-1 flex items-center justify-center text-zinc-400/60'>
              <span className='text-base text-center gap-2'>
                <BirdIcon className='size-5 inline' /> Você está conversando
                sobre <span className='underline'>{title}</span>. <br /> Envie
                uma mensagem para que a IA responda
              </span>
            </div>
          ) : (
            chats.map(({ from, message }, index) => {
              return (
                <div
                  key={index}
                  className={`rounded-2xl ${
                    from == 'human' ? 'ml-auto' : ''
                  } py-2.5 px-4 max-w-96 w-fit h-fit bg-zinc-100`}
                >
                  {message}
                </div>
              )
            })
          )}

          {loading && (
            <span className='animate-pulse'>
              ✨ procurando pela melhor resposta...
            </span>
          )}
        </div>

        <div className='fixed bottom-0 w-full max-w-2xl p-4'>
          <form
            onSubmit={event => {
              event.preventDefault()

              const message = (
                event.currentTarget.firstChild as HTMLInputElement
              ).value
              startPrompt({
                message
              })

              event.currentTarget.reset()
            }}
            className='relative bg-white shadow-lg rounded-2xl'
          >
            <input
              name='human-message'
              type='text'
              placeholder='Enviar mensagem...'
              className='w-full p-4 pr-12 rounded-2xl border border-gray-200 outline-none focus:border-gray-300 focus:ring-2 focus:ring-gray-100 transition-all'
            />
            <button
              type='submit'
              className='absolute right-4 top-1/2 -translate-y-1/2 p-2 hover:bg-gray-50 rounded-lg transition-colors text-gray-400 hover:text-gray-600 bg-slate-200/40 px-2.5'
            >
              ↑
            </button>
          </form>
        </div>
      </main>
    </div>
  )
}
