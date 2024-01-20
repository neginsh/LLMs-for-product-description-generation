import numpy as np
import pandas as pd
import pprint
import requests
from transformers import pipeline,set_seed,Conversation
import logging
import translators as ts
import xmltodict
from selectorlib import Extractor
import random
import language_tool_python
import nltk
import readability
import syntok.segmenter as segmenter
import spacy
import time
from ast import literal_eval
import re

class Description_generator:

    def __init__(self):
        self.load_LLM()
        return
        
    def load_LLM(self):
        self.pipe = pipeline("text-generation", model="malteos/bloom-6b4-clp-german")

    def generate_prompt(self,product,nr_of_shots,category_description = "",shots=[]):
        message = ""

        if category_description != "":
            category_description ="Katagoriebeschreibung: "+category_description+ ",\n"
        else:
            shots = self.shots_modified(nr_of_shots=nr_of_shots,shots=shots)


        if nr_of_shots == 0 :
            message = f"""
            Schreib nun die Produktbeschreibung für dieses Produkt.
            Produktname: {product.Webbezeichnung},\n
            Produktkategorie: {product.ECLASS_Name},\n
            Marke: {product.Marke},\n
            Hersteller: {product.Hersteller},\n
            {category_description if category_description != "" else ""}
            Produktbeschreibung:"""
        elif nr_of_shots == 1:
            message = f"""
            Schreib die Produktbeschreibung für das folgende Produkt. Hier ist ein Beispiel:
            {shots[0]} \n
            Schreib nun die Produktbeschreibung für dieses Produkt.
            Produktname: {product.Webbezeichnung},\n
            Produktkategorie: {product.ECLASS_Name},\n
            {category_description if category_description != "" else ""}
            Produktbeschreibung:"""
        elif nr_of_shots == 2:
            message = f"""
            Schreib die Produktbeschreibung für das folgende Produkt. Hier ist zwei Beispiele:
            {shots[0]} \n
            {shots[1]} \n
            Schreib nun die Produktbeschreibung für dieses Produkt.
            Produktname: {product.Webbezeichnung},\n
            Produktkategorie: {product.ECLASS_Name},\n
            {category_description if category_description != "" else ""}
            Produktbeschreibung:"""

        return message

    def shots_modified(self,nr_of_shots,shots):
        for i in range(nr_of_shots):
            shot = shots[i]
            category_description = shot[shot.index("Katagoriebeschreibung:"):shot.index("Produktbeschreibung")]
            shots[i] = shot.replace(category_description,"")

        return shots
    
    def generate(self,product,nr_of_shots,category_description = "",shots=[], temperature = 0.8):
        message = self.generate_prompt(product=product,nr_of_shots=nr_of_shots,category_description=category_description,shots=shots)
        l = len(message)
        output = self.pipe(message,max_new_tokens= 150,max_length = l+150,num_return_sequences= 1, do_sample = True,temperature = temperature,repetition_penalty =4.0)[0]['generated_text']
        return output[output.index("Produktbeschreibung:")+len("Produktbeschreibung:"):]



class preprocessing:

    def translate(self,text,from_lang,to_lang):
        return ts.translate_text(text, from_language=from_lang, to_language=to_lang, translator = 'google')
    
    def generate_category_description(self,eclass):# using dbpedia

        if "(" in eclass:
            category = self.translate(text = eclass[0:eclass.index("(")], from_lang='de', to_lang='en')
        else:
            category = self.translate(text = eclass, from_lang='de', to_lang='en')

        response = requests.get("https://lookup.dbpedia.org/api/search?query="+ category,
                        headers ={'Content-Type': 'application/json'},
                        params = {"lang":"de"})

        data_dict = xmltodict.parse(response.text)
        for i in range(0,len(data_dict["ArrayOfResults"]['Result'])):
            desription = data_dict["ArrayOfResults"]['Result'][i]['Description'].lower()
            if category.lower() in desription[0:desription.index("is")]:
                # should be in the first half before "is"
                return self.translate(category, from_lang='en', to_lang='de')
            
    def scrape(self,url,e):  
        
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0',
            'dnt': '1',
            'upgrade-insecure-requests': '1',
            # 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-user': '?1',
            'sec-fetch-dest': 'document',
            'referer': 'https://www.amazon.de/',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        }
        # url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=de&ssl=all&anonymity=all"
        # r = requests.request("GET", url)
        # ip = random.choice(r.text.split("\r\n"))
        # proxy = {
        #     'http': ip,
        #     'https': ip,
        # }

        # Download the page using requests
        print("Downloading %s"%url)
        with requests.Session() as s:
            r = s.get(url, headers=headers)
        # Simple check to check if page was blocked (Usually 503)
        if r.status_code > 500:
            if "To discuss automated access to Amazon data please contact" in r.text:
                print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
            else:
                print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
            return None
        # Pass the HTML of the page and create 
        return e.extract(r.text)
    
    def get_2_ex_from_amazon(self,eclass):
        e_search = Extractor.from_yaml_file('search_results.yml')
        r = self.scrape("https://www.amazon.de/s?k="+eclass+"&__mk_de_DE=ÅMÅŽÕÑ&ref=sr_st_relevancerank&",e_search)
        shots = []
        x = r["products"][0]["url"]
        start = "/dp/"
        shots[0] = x[x.index(start)+len(start):x.index("/",x.index(start)+len(start))]
        x = r["products"][1]["url"]
        shots[1] = x[x.index(start)+len(start):x.index("/",x.index(start)+len(start))]
        e_detail = Extractor.from_yaml_file('selectors.yml')
        shots[0] = self.scrape("https://www.amazon.de/dp/"+shots[0],e_detail)
        shots[1] = self.scrape("https://www.amazon.de/dp/"+shots[1],e_detail)
        return shots


    def generate_shots(self,eclass,category_description):
        shots = self.get_2_ex_from_amazon(eclass)
        shots[0] = f"""Produktname: {shots[0]["name"]},\n
                    Produktkategorie: {eclass},\n
                    Katagoriebeschreibung: {category_description},\n
                    Produktbeschreibung : {shots[0]["short_description"]}
                    """
        shots[1] = f"""Produktname: {shots[1]["name"]},\n
                    Produktkategorie: {eclass},\n
                    Katagoriebeschreibung: {category_description},\n
                    Produktbeschreibung : {shots[1]["short_description"]}
                    """
        return shots

        


class PostProcessing:

    def remove_hellucination(self,text):
        longest_word = "Rinderkennzeichnungsfleischetikettierungsüberwachungsaufgabenübertragungsgesetz"
        for w in text.split():
            if len(w) > len(longest_word):
                text = text[:text.find(w)]
            break
        
        return text
    
    def grammar_fix(self,text):
        tool = language_tool_python.LanguageToolPublicAPI('de')
        is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()
        matches = tool.check(text)
        matches = [rule for rule in matches if not is_bad_rule(rule)]
        text = language_tool_python.utils.correct(text, matches)
        return text
    
    def rewrite(self,text,product,pipe):
        # print("rewrite prompt:")
        p = """Überarbeiten Sie die Produktbeschreibung, um sicherzustellen, dass sie frei von Halluzinationen ist, während wichtige Informationen über das Produkt erhalten bleiben. 
            \n Produktbeschreibung: """+text+"""\n Überarbeiteter Text:"""
        # print(p)
        l = len(p)
        output = pipe(p,max_new_tokens= 64,num_return_sequences= 1, do_sample = True,temperature = 0.4,repetition_penalty =4.0)[0]['generated_text']
        return output[output.index("Text:")+len("Text:"):]
    
    def postprocess(self,text,product,pipe):
        text = self.remove_hellucination(text)
        # text = self.grammar_fix(text)
        text = self.rewrite(text,product,pipe)
        return text
    

class Evaluation:
    def __init__(self):
            self.nlp = spacy.load("de_core_news_md")

            
    def translate(self,text,from_lang,to_lang):
        return ts.translate_text(text, from_language=from_lang, to_language=to_lang)

    def readability(self,text):
        if text == 0 or text == None:
            return None
        tokenized = '\n\n'.join(
            '\n'.join(' '.join(token.value for token in sentence)
                for sentence in paragraph)
            for paragraph in segmenter.analyze(text))
        
        results = readability.getmeasures(tokenized, lang='de')
        # return results['readability grades']
        return results
    
    def blue(self,hypothesis,reference):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
        return BLEUscore
    
    def class_classification(self,text):
        API_URL = "https://api-inference.huggingface.co/models/deutsche-telekom/bert-multi-english-german-squad2"
        headers = {"Authorization": "Bearer ********"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query({
            "inputs": {
            "question": "Zu welcher Kategorie gehört dieses Produkt?",
            "context": text
            },
        })
        while "error" in output:
            time.sleep(5)
            output = query({
                "inputs": {
                "question": "Zu welcher Kategorie gehört dieses Produkt?",
                "context": text
                },
            })

        return output["answer"]
    

    def get_word_embedding_safe(self,word):
        """
        Get the word embedding vector for a given word.
        Return None if the word is not in the vocabulary.
        """
        try:
            return self.nlp(word).vector
        except KeyError:
            print(f"Word '{word}' not in vocabulary.")
            return None

    # Modify compare_similarity function
    def compare_similarity(self,word1, word2):
        """
        Compare the similarity between two words.
        """
        vec1 = self.get_word_embedding_safe(word1)
        vec2 = self.get_word_embedding_safe(word2)

        if vec1 is not None and vec2 is not None:
            similarity = self.cosine_similarity(vec1, vec2)
            return similarity
        else:
            return None

    def cosine_similarity(self,vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
        similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
        return similarity
    
    def get_generalized_representation(self,words):
        vectors = [self.nlp(word).vector for word in words if self.nlp.vocab[word].has_vector]
        if vectors:
            # Calculate the average vector
            avg_vector = sum(vectors) / len(vectors)
            return avg_vector
        else:
            return None 
    
    def cc_metric(self,text,actual_class):
        c = self.class_classification(text)
        en_c = self.translate(c,"de","en")
        en_c = en_c.split(" ")
        de_c = [self.translate(word,"en","de") for word in en_c]
        en_ac = self.translate(actual_class,"de","en")
        en_ac = en_ac.split(" ")
        de_ac = [self.translate(word,"en","de") for word in en_ac]
        vec_c = self.get_generalized_representation(de_c)
        vec_ac = self.get_generalized_representation(de_ac)
        if vec_c is None:
            return 0
        return self.cosine_similarity(vec_c,vec_ac)
    
    def cc_pbs(self,text):
        if text == 0 or text == None:
            return None
        
        url = 'http://127.0.0.1:5000/pbs-inference/eclass-for-text'
        data = {'text': text}
        x = requests.post(url, json = data)

        predicted_eclass = literal_eval(x.text)["eclass"]
        return predicted_eclass
        # print(predicted_eclass) 
        # print(actual_class)
    def cc_pbs_metric(self,predicted_eclass,actual_class,level):

        if level == 4:
            return str(predicted_eclass) == str(actual_class)
        elif level == 3:
            return str(predicted_eclass)[:6] == str(actual_class)[:6]
        elif level == 2:
            return str(predicted_eclass)[:4] == str(actual_class)[:4]
        elif level == 1:
            return str(predicted_eclass)[:2] == str(actual_class)[:2]
        else:
            return 0
        

    def coherence_metric(self,pipe,text):
        p = """Sie erhalten eine für ein Produkt verfasste Produktbeschreibung.

            Ihre Aufgabe ist es, die Produktbeschreibung anhand eines Kriteriums zu bewerten.

            Bitte stellen Sie sicher, dass Sie diese Anweisungen sorgfältig lesen und verstehen. Halten Sie dieses Dokument offen, während Sie die Bewertung vornehmen, und konsultieren Sie es bei Bedarf.

            Bewertungskriterien:

            Kohärenz (1-5) - die Gesamtqualität aller Sätze. Wir orientieren uns bei dieser Dimension an der DUC-Qualitätsfrage nach Struktur und Kohärenz, wobei "die Produktbeschreibung gut strukturiert und gut organisiert sein sollte. Die Produktbeschreibung sollte nicht nur ein Haufen zusammenhängender Informationen sein, sondern von Satz zu einem kohärenten Informationskörper zu einem Thema aufbauen."

            Bewertungsschritte:

            Lesen Sie die Produktbeschreibung sorgfältig durch und identifizieren Sie die Produktkategorie und die wichtigsten Produktmerkmale.
            Überprüfen Sie, ob die Produktbeschreibung diese in einer klaren und logischen Reihenfolge präsentiert.
            Weisen Sie der Kohärenz auf einer Skala von 1 bis 5 einen Punktwert zu, wobei 1 das niedrigste und 5 das höchste ist, basierend auf den Bewertungskriterien.
            Beispiel:

            Produktbeschreibung:

            """+ text +"""

            Bewertungsformular (nur Punktzahlen):

            Kohärenz:"""
        
        l = len(p)
        output = pipe(p,max_new_tokens= 1,num_return_sequences= 1, do_sample = True,temperature = 0.8,repetition_penalty =4.0)[0]['generated_text']
        print(output[output.index("Kohärenz:")+len("Kohärenz:"):])
        score = re.findall(r'\d+',output[output.index("Kohärenz:")+len("Kohärenz:"):])
        if len(score) == 0:
            return None
        return score[0]





def generate(row,is_with_category_description,nr_of_shots,dg):
    global i
    print(i)
    i +=1
    

    if row.ECLASS_8_1 in pp.ECLASS_8_1.values :
        category_description = pp.loc[pp.ECLASS_8_1 == row.ECLASS_8_1].category_description.values[0]
        shots = pp.loc[pp.ECLASS_8_1 == row.ECLASS_8_1].shots.values[0]
        if is_with_category_description and category_description == None:
            return None
        if nr_of_shots != 0 and shots != None:
            shots = literal_eval(shots)
        elif nr_of_shots != 0 and shots == None:
            return None
        return dg.generate(product=row,nr_of_shots=nr_of_shots,category_description=category_description,shots=shots)
    return None

def postprocess(row,post,dg):
    global i
    print(i)
    i +=1
    if row.ECLASS_8_1 in pp.ECLASS_8_1.values :
        if row.description == None:
            return None
        text = post.postprocess(row.description,row,dg.pipe)
            
        return text
    return None


def generate_for_all(sample,pp,df,is_with_category_description,nr_of_shots):
    global i
    prep = preprocessing()
    dg = Description_generator()
    post = PostProcessing()
    ev = Evaluation()
    
    print("description generation")
    sample["description"] = 0
    sample["description"] = sample.apply(lambda x: generate(x,is_with_category_description,nr_of_shots,dg),axis=1)

    print("post processing")
    i = 0
    sample["postprocessed"] = 0
    sample["postprocessed"] = sample.apply(lambda x: postprocess(x,post,dg),axis=1) 

    sample= sample.fillna(np.nan).replace([np.nan], [None])

    print("evaluation readibility")

    sample["FleschReadingEase"] = 0
    sample["FleschReadingEase"] = sample.apply(lambda x: ev.readability(x.postprocessed)['readability grades']['FleschReadingEase'] if x.postprocessed != None else None,axis=1)

    
    sample["complex_word_count"] = 0
    sample["complex_word_count"] = sample.apply(lambda x: ev.readability(x.postprocessed)['sentence info']['complex_words_dc'] if x.postprocessed != None else None,axis=1)

    
    print("evaluation cc")

    sample["pbs_cc_level1"] = 0
    sample["pbs_cc_level2"] = 0
    sample["pbs_cc_level3"] = 0
    sample["pbs_cc_level4"] = 0
    sample["pbs_cc"] = 0
    sample["pbs_cc"] = sample.apply(lambda x: ev.cc_pbs(x.postprocessed),axis=1)
    sample["pbs_cc_level1"] = sample.apply(lambda x: ev.cc_pbs_metric(x.pbs_cc,x.ECLASS_8_1,1) if x.postprocessed != None else None,axis=1)
    sample["pbs_cc_level2"] = sample.apply(lambda x: ev.cc_pbs_metric(x.pbs_cc,x.ECLASS_8_1,2) if x.postprocessed != None else None,axis=1)
    sample["pbs_cc_level3"] = sample.apply(lambda x: ev.cc_pbs_metric(x.pbs_cc,x.ECLASS_8_1,3) if x.postprocessed != None else None,axis=1)
    sample["pbs_cc_level4"] = sample.apply(lambda x: ev.cc_pbs_metric(x.pbs_cc,x.ECLASS_8_1,4) if x.postprocessed != None else None,axis=1)

    # sample["bert_cc"] = 0
    # sample["bert_cc"] = sample.apply(lambda x: ev.cc_metric(x.postprocessed,x.ECLASS_8_1),axis=1)

    return sample


i = 0
df = pd.read_csv('./data.csv',sep = ";", encoding='utf-8-sig')
df_classes = pd.read_csv('./eClass8.1_Namen.csv',sep = ",", encoding='utf-8-sig')
pp = pd.read_csv('./pp.csv',sep = ";", encoding='utf-8-sig',dtype={'ECLASS_8_1': 'Int32'}).set_index("Unnamed: 0").fillna(np.nan).replace([np.nan], [None])
# pp["shots"] = """['Produktname: Element System 10700-00027 10700 STAHLFACHBODEN - Regalboden für Wandschiene und Pro-Regalträger, Stahl, Weiß, 800x350, 2 Stück,                    Produktkategorie: Regal, Regalsystem (Büroeinrichtung),                    Katagoriebeschreibung: Regalset, kombiniert zu einem Möbelstück,                    Produktbeschreibung :  Die Regale von DIY ELEMENT SYSTEM eignen sich hervorragend für alle Wohn- und Lebensbereiche und lassen sich individuell an jeden Raum anpassen. Die Regalböden werden einfach auf die Pro-Träger aufgelegt. VIELSEITIG – Ideal für Hobbyraum, Garage oder Speisekammer. Durch seine abwaschbare und robuste Oberfläche ist der Stahlfachboden vielseitig einsetzbar und dank der Tragkraft finden auch schwere Gegenstände einen sicheren Platz. DIE RICHTIGE WAHL – Unser umfangreiches Sortiment überzeugt mit hervorragender Qualität Made in Germany und garantiert jahrelange Freude an den gekauften Produkten. TECHNISCHE DETAILS - 800x350; Material Stahl; FarbeWeiß; LIEFERUMFANG - 2 Stk. Stahlfachboden Regalboden 10700-00027 für Wandschiene und Pro-Regalträger                     ', 'Produktname: SONGMICS Würfelregal, Kleiderschrank Kunststoff, Steckregal groß, Regalsystem mit Türen und Kleiderstangen, jedes Fach 35 x 35 x 35 cm, für Schlafzimmer, tintenschwarz LPC301B01,                    Produktkategorie: Regal, Regalsystem (Büroeinrichtung),                    Katagoriebeschreibung: Regalset, kombiniert zu einem Möbelstück,                   Produktbeschreibung :  [Viel Stauraum] Bringen Sie Ordnung in Ihren Raum mit diesem großen 36 x 143 x 178 cm Würfelregal. Mit zwölf 35 x 35 x 35 cm großen Würfeln und 2 Kleiderstangen haben Sie Platz für Kleidung, Schuhe, Bettwäsche und vieles mehr [Hochwertig & sicher] Hochwertige PP-Kunststoffplatten mit Metallrahmen und langlebigen ABS-Kunststoffverbindern lassen das Regalsystem bis 130 kg insgesamt und 10 kg pro Würfel tragen, während die mitgelieferten Kippschutzteile für Stabilität sorgen [Multifunktional für jeden Raum] Verwenden Sie dieses vielseitige Steckregal als Schuhschrank im Eingangsbereich, als Schrank in Ihrem Schlafzimmer oder Ankleidezimmer, um Ihre Kleidung, Taschen und Accessoires zu organisieren [Mühelos zu reinigen] Dieser feuchtigkeitsresistente Kunststoffschrank mit staubdichten Türen hält Ihre Kleidung trocken und sauber und kann mit einem feuchten Tuch gereinigt werden, was die tägliche Pflege leicht und schnell macht [Leichter Aufbau] Dieser Aufbewahrungsschrank lässt sich mit dem mitgelieferten Gummihammer leicht auf- und abbauen, sodass Sie seine Form ganz einfach an Ihren Raum anpassen können                   ']"""
sample = pd.read_csv('./sample.csv',sep = ";", encoding='utf-8-sig')
i = 0
sample = generate_for_all(sample,pp,df,True,0)
sample.to_csv('./sample_with_0.csv',sep = ";", encoding='utf-8-sig', index=False)

# for j in range(1,3):
#     sample = pd.read_csv('./sample.csv',sep = ";", encoding='utf-8-sig')
#     # sample = sample.dropna()
#     i = 0
#     sample = generate_for_all(sample,pp,df,False,j)
#     sample.to_csv('./sample_same_category_shot_with_'+str(j)+'.csv',sep = ";", encoding='utf-8-sig', index=False)



