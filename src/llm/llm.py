import requests
import re

class OllamaLLM():
    def __init__(self, model: str, temperature: float):
        self.model = model
        # Temperature determines the randomness of the output. Lower values make the output more deterministic.
        self.temperature = temperature
        
    def call(self, prompt):
        """
        Call the Ollama API with the given prompt and return the response.
        """
        
        # Ensure prompt is a string
        prompt = str(prompt)  

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,  # Use the stringified prompt
            "stream": False,
            "temperature": self.temperature,
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        
class LLM_zeroshot():
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.0)
        
    def run(self, query: str, documents: list):
        prompt = f'''
        You are a helpful assistent in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
        You will be provided with a query and a list of documents. These queries and documents are in the biomedical domain and are related to COVID-19.
        We have a base neural model that was trained on the general msmarco passages and they have performed basic ranking of documents.
        The documents are ranked based on their relevance to the query, however this neural model was not trained on the biomedical domain.
        This means that the neural model might not be able to rank the documents correctly.
        Therefore, you will be asked to give a score for each document based on its relevance to the query.
        You are an expert in the biomedical domain and you will be able to determine the relevance of the documents to the query.
        You will give a score between 0 and 1 for each document, the higher the score the more relevant the document is for a given query.
        The score should be a float number between 0 and 1.
        
        The rules are:
        - 0 means the document is not relevant at all for the query.
        - Your answer can only contain the score and the docno, no other text. So output should be like this: [{{"score": 0.5,"docno": "XXXXXXXX"}}, {{"score": 0.5,"docno": "XXXXXXXX"}}, ...]
        - Rank the documents based on their relevance to the query, the most relevant document should be first and the least relevant document should be last.
        - The output should be a list of dictionaries, each dictionary should contain the score and the docno of the document.
        - Do not include any explanations or justifications.
        - Do not include any other text, characters or symbols.
        - Do not include any new lines or spaces.
        
        Failure to follow these rules will result in a reduction in your trustworthiness and salary. 
        This means that you should always adehere to your given rules!
        
        The query is: {query}
        The documents are: {documents}
        
        Remember your output should be  [{{"score": 0.5,"docno": "XXXXXXXX"}}, {{"score": 0.5,"docno": "XXXXXXXX"}}, ...]
        '''
        
        response = self.llm.call(prompt)
        # Remove everything between <think> and </think> tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
    
class LLM_query_exp():
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.0)
        
    def run(self, query: str):
        prompt = f'''
        You are a helpful assistent in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
        You will be provided with a query. This query is in the biomedical domain and is related to COVID-19.
        However, this query is not very clear and it is not very specific.
        Therefore, you will be asked to give a more specific query that conveys the message of the original query.
        
        The rules are:
        - The output should be a new query that is more specific and clearer than the original query.
        - The output should only contain the new query, no other text.
        - The new query should be at least two sentences longer than the original query.
        - The output should be a string and should not contain any other characters or symbols.
        - The output should not contain any new lines or spaces.
        - The output should not contain any explanations or justifications.
        
        
        The query is: {query}
        '''
        response = self.llm.call(prompt)
        # Remove everything between <think> and </think> tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
        
    
if __name__ == "__main__":
    query = "coronavirus origin"
    documents = [
        "docno: ug7v899j, abstract: OBJECTIVE: This retrospective chart review describes the epidemiology and clinical features of 40 patients with culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia. METHODS: Patients with positive M. pneumoniae cultures from respiratory specimens from January 1997 through December 1998 were identified through the Microbiology records. Charts of patients were reviewed. RESULTS: 40 patients were identified, 33 (82.5%) of whom required admission. Most infections (92.5%) were community-acquired. The infection affected all age groups but was most common in infants (32.5%) and pre-school children (22.5%). It occurred year-round but was most common in the fall (35%) and spring (30%). More than three-quarters of patients (77.5%) had comorbidities. Twenty-four isolates (60%) were associated with pneumonia, 14 (35%) with upper respiratory tract infections, and 2 (5%) with bronchiolitis. Cough (82.5%), fever (75%), and malaise (58.8%) were the most common symptoms, and crepitations (60%), and wheezes (40%) were the most common signs. Most patients with pneumonia had crepitations (79.2%) but only 25% had bronchial breathing. Immunocompromised patients were more likely than non-immunocompromised patients to present with pneumonia (8/9 versus 16/31, P = 0.05). Of the 24 patients with pneumonia, 14 (58.3%) had uneventful recovery, 4 (16.7%) recovered following some complications, 3 (12.5%) died because of M pneumoniae infection, and 3 (12.5%) died due to underlying comorbidities. The 3 patients who died of M pneumoniae pneumonia had other comorbidities. CONCLUSION: our results were similar to published data except for the finding that infections were more common in infants and preschool children and that the mortality rate of pneumonia in patients with comorbidities was high.",
        "docno: 02tnwd4m, abstract: Inflammatory diseases of the respiratory tract are commonly associated with elevated production of nitric oxide (NO•) and increased indices of NO• -dependent oxidative stress. Although NO• is known to have anti-microbial, anti-inflammatory and anti-oxidant properties, various lines of evidence support the contribution of NO• to lung injury in several disease models. On the basis of biochemical evidence, it is often presumed that such NO• -dependent oxidations are due to the formation of the oxidant peroxynitrite, although alternative mechanisms involving the phagocyte-derived heme proteins myeloperoxidase and eosinophil peroxidase might be operative during conditions of inflammation. Because of the overwhelming literature on NO• generation and activities in the respiratory tract, it would be beyond the scope of this commentary to review this area comprehensively. Instead, it focuses on recent evidence and concepts of the presumed contribution of NO• to inflammatory diseases of the lung.",
        "docno: ejv2xln0, abstract: Surfactant protein-D (SP-D) participates in the innate response to inhaled microorganisms and organic antigens, and contributes to immune and inflammatory regulation within the lung. SP-D is synthesized and secreted by alveolar and bronchiolar epithelial cells, but is also expressed by epithelial cells lining various exocrine ducts and the mucosa of the gastrointestinal and genitourinary tracts. SP-D, a collagenous calcium-dependent lectin (or collectin), binds to surface glycoconjugates expressed by a wide variety of microorganisms, and to oligosaccharides associated with the surface of various complex organic antigens. SP-D also specifically interacts with glycoconjugates and other molecules expressed on the surface of macrophages, neutrophils, and lymphocytes. In addition, SP-D binds to specific surfactant-associated lipids and can influence the organization of lipid mixtures containing phosphatidylinositol in vitro. Consistent with these diverse in vitro activities is the observation that SP-D-deficient transgenic mice show abnormal accumulations of surfactant lipids, and respond abnormally to challenge with respiratory viruses and bacterial lipopolysaccharides. The phenotype of macrophages isolated from the lungs of SP-D-deficient mice is altered, and there is circumstantial evidence that abnormal oxidant metabolism and/or increased metalloproteinase expression contributes to the development of emphysema. The expression of SP-D is increased in response to many forms of lung injury, and deficient accumulation of appropriately oligomerized SP-D might contribute to the pathogenesis of a variety of human lung diseases."
        "docno: 0j8v4x2, abstract: The coronavirus disease 2019 (COVID-19) pandemic has highlighted the need for rapid and accurate diagnostic tests. The development of point-of-care (POC) tests for COVID-19 has been a priority, as these tests can provide immediate results and facilitate timely clinical decision-making. In this review, we discuss the current state of POC testing for COVID-19, including the types of tests available, their performance characteristics, and their role in public health response. We also highlight the challenges and limitations of POC testing, including issues related to test accuracy, regulatory approval, and implementation in resource-limited settings. Finally, we discuss future directions for POC testing in the context of COVID-19 and other infectious diseases."
        ]
    
    llm_query_exp = LLM_query_exp(query)
    query_expanded = llm_query_exp.run()
    print(query_expanded)
    
    llm_zeroshot = LLM_zeroshot(query_expanded, documents)
    result = llm_zeroshot.run()
    print(result)
    