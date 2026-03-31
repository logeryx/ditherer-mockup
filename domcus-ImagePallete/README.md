# Documentation of the task "02-ColorPallete"

## Author
Dominika Plačková

## Command line arguments
| short | long | description | mandatory |
| -------- | -------- | ----------------------------- | -------- |
| -i \<int\> | --input \<int\> | Input image filename | Yes |
| -c \<int\> | --colors \<int\> | Number of colors on pallete | No |
| -o \<filename\> | --output \<filename\> | Output file-name (*.png) | No |

## Input data
Povolene formáty vstupných súborov sú JPG a PNG. Ak je zadaný iný program skončí errorom.
Povolený formát výstupného súboru je PNG. Pri zadaní iného program skončí errorom. 

## Algorithm
Hlavný algoritmus kvantizácie farieb je Median Cut implementovaný v metódach MedianCutOnHistogram a pomocných triedach RgbWeighted a ColorBoxWeighted.

Princíp Fungovania
Median Cut funguje na princípe rekurzívneho delenia farebného priestoru (v tomto prípade RGB priestoru histogramu):
1. Vytvorenie Boxu: Všetky farby (vážené ich výskytom) sú obsiahnuté v jednom boxe.
2. Výber Boxu na Delenie: V každej iterácii (cyklus while (boxes.Count < targetCount)) sa vyberie ten box, ktorý má najväčší rozsah (Range). Rozsah je definovaný ako maximálna vzdialenosť medzi minimálnou a maximálnou farebnou zložkou (R, G alebo B).
3. Výber Osi Delenia: Box sa delí pozdĺž tej osi (R, G, alebo B), ktorá má najväčší rozsah (napr. ak je najväčší rozptyl farieb v modrej zložke, delí sa os B).
4. Delenie v Mediáne: Pixely v boxe sa zoradia podľa hodnoty vybranej zložky (napr. B) a box sa rozdelí presne v mediáne počtu pixelov (nie mediáne hodnoty farby). Tým sa zabezpečí, že každý nový pod-box obsahuje približne rovnaký počet pixelov (hustotu).
5. Výsledok: Proces sa opakuje, kým sa nedosiahne požadovaný počet klastrov (targetCount). Reprezentatívna farba každého klastra je určená ako vážený priemer všetkých pixelov v ňom (AverageColor).

## Extra work / Bonuses
Farby sú triedené podľa kombinovaného kľúča, ktorý uprednostňuje spektrálne poradie (dúha), ale jemne zohľadňuje aj jas (svetlosť) Kód prevedie farbu z RGB do HSV (pomocou RgbToHsv). 
Primárne sa farby zoradia podľa hodnoty Hue, čím sa dosiahne hladký prechod Červená -> Žltá -> Zelená -> Modrá -> Fialová.
Sekundárne triedenie (V - Value/Jas): K hodnote Hue sa pripočíta váha. Ak existuje viacero odtieňov modrej, budú zoradené od najtmavšej po najsvetlejšiu
Vykreslenie palety je fo forme, obrázka, ktorý obsahuje daný obrázok a pod ním jeho paletu.

## Use of AI
Pri volbe algoritmu som zo začiatku pracovala s K-means. Avšak tento prístup sa mi neosvedčil, kvôli tomu, že farby na malej ploche
bola zanedbávaná. V hľadaní lepšieho pístupu mi pomohol ChatGPT, kotrý mi poradil MedianCut, ktorý som použila aj v mojom riešení.
Median Cut zaručí, že sa do klastrov rozdelia najprv také farby, ktoré sú najviac vizuálne odlišné.
Pri vykresľovní palety som pre esteticky príjemnejší dojem, usporiadala farby podľa Hue a sekundárne podľa Jasu. Toto usporiadanie, my
taktiež odporúčilo AI, po niekoľko pokusov skúšania iných farebných systémov ako RGB alebo LAB, som najviac bola spokjná s týmto. Všimla som si, 
že modrá a zelená farba majú tendenciu byť ak sú na malej ploche viac zanedbávané ako napríklad červená a oranžová.
AI mi napísalo celú logiku, čo stála za spradcovaním obrázka a následným vytvorením palety. 

https://chatgpt.com/g/g-p-68ed4e782ed081919008fdfda2ea124e-pc-grafika/shared/c/68ef9aa4-dfd4-8326-9382-825dc578252e?owner_user_id=user-79eciDhz4hIhfjnWwhjypPy9
