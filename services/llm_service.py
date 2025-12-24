from groq import Groq
from dotenv import load_dotenv
import os
from core.patient import Patient

load_dotenv()

class LLMService:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY")) 

    def generate_medicine_query(self, patient_data: dict):
        first_response = self.client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
                {
                "role": "system",
                "content": (
                    '''Você é um motor de busca biomédico. Sua tarefa é converter um quadro clínico em uma lista de termos técnicos para uma busca vetorial em uma base de dados de medicamentos.

                    **Objetivo:** Gerar uma string, separada por vírgulas, contendo os termos mais relevantes para encontrar medicamentos adequados.

                    **O que incluir na sua resposta:**
                    - Classes terapêuticas relevantes (ex: "analgésico", "antipirético", "anti-inflamatório não esteroide", "anti-histamínico").
                    - Princípios ativos potencialmente indicados (ex: "ibuprofeno", "loratadina").
                    - Mecanismos de ação, se aplicável (ex: "inibidor da COX-2").
                    - Sintomas-chave a serem tratados (ex: "tosse produtiva", "congestão nasal", "dor de cabeça tensional").

                    **Regras CRÍTICAS:**
                    1.  **NUNCA** inclua na sua lista qualquer termo (princípio ativo, classe, etc.) que esteja presente nas 'Alergias' ou 'Reações adversas passadas' do paciente.
                    2.  Sua saída deve ser **APENAS** a lista de termos, separados por vírgula. Não adicione frases como "Com base no quadro, os termos são:" ou qualquer outra explicação.

                    **Exemplo de saída desejada para um paciente com febre e dor no corpo, sem alergias:**
                    analgésico, antipirético, anti-inflamatório, ibuprofeno, dor de cabeça, febre, dor muscular
                    '''
                    )
                },
                {
                "role": "user",
                "content": patient_data
                }
            ]
        )
        return first_response.choices[0].message.content

    def generate_final_medicine_suggestion(self, patient_data: dict, docs, initial_analysis):
        if not docs:
            return "Com base na análise inicial, não foi encontrado nenhum medicamento seguro no banco de dados que não esteja associado às alergias ou reações adversas do paciente. Recomenda-se uma avaliação médica mais aprofundada."

        docs_text = "\n---\n".join([doc.page_content for doc in docs])

        #Faz nova pergunta à LLM, agora com contexto do RAG
        final_prompt = (
            f"{patient_data}\n"
            f"Análise clínica inicial da IA:\n{initial_analysis}\n\n"
            f"Lista de medicamentos encontrados com base nessa análise:\n{docs_text}\n\n"
            f"Com base no histórico clínico do paciente, nas alergias e reações passadas, e nos medicamentos encontrados, indique as melhores opções, justificando clinicamente sua recomendação. "
            f"Evite sugerir medicamentos com risco potencial. Use linguagem acessível."
        )

        final_response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é uma IA médica confiável. Avalie cuidadosamente os dados clínicos e os medicamentos encontrados e dê sugestões seguras e bem fundamentadas. "
                        "Evite qualquer menção a medicamentos que o paciente não deve utilizar."
                    )
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ]
        )

        print("Documentos retornados pelo RAG:\n", docs_text)
        return final_response.choices[0].message.content

    def generate_exam_suggestion(self, patient_data: dict):
        final_response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é uma IA médica confiável. Avalie cuidadosamente os dados clínicos e os medicamentos encontrados e dê sugestões seguras e bem fundamentadas de exames que o paciente poderia fazer. "
                        "Evite qualquer menção a medicamentos que o paciente não deve utilizar."
                    )
                },
                {
                    "role": "user",
                    "content": patient_data
                }
            ]
        )
        return final_response.choices[0].message.content