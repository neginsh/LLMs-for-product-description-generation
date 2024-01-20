# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="jphme/em_german_mistral_v01")
p = 'Schreiben Sie eine Produktbeschreibung für das folgende Produkt: \n Produktname: Lasertoner cyan OKI 42804547 \n Produktkategorie: Toner, Tonereinheit (Laserdrucker, Kopierer)\n'
output = pipe(p,max_new_tokens= 150)[0]['generated_text']
print(output)