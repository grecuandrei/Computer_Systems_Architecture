# Tema 3 - GPU Hashtable

# Autor: Grecu Andrei-George

## Cum s-a implementat soluția ?

	Initial am setat fiecare cheie din hashmap-ul creat ca fiind 0, pentru
a implementa logica programului.

	Am folosit conceptul de linear probing, si am creat un hashmap ca un
vector de valori de tip asociere (cheie, valoare).

	Linear Probing -> pentru fiecare entry pe care vreau sa-l inserez,
sau pentru fiecare cautare efectuata, se va calcula pozitia folosind functia
de hash. Daca cheia nu este 0 si nu se face un match, se trece la urmatoarea pozitie
si se verifica.

	Insert -> se va face o verificare privind dimensiunea de elemente care
incap in hashmap. Daca nu exista destul de mult loc se va face reshape. 
Valoare pentru load factor am ales sa o declar ca fiind 80%. Pentru functia
gpu_kernel_insert, se va apela kernel_insert cu parametrii corespunzatori.
Se va parcurge vectorul in care se va executa inserarea. In functia de hash
se va calcula pozitia pe care se va insera asocierea (cheie, valoare). Verific
daca pozitia obtinuta are vreo valoare, daca key = 0, atunci se va realiza 
inserarea.

	Get-> se va crea un vector de valori care va fi transmis ca parametru
pentru functia kernel. Se va retine in vectorul deviceValues, valorile
pe care le gasesc pe pozitia determinata de functia de hash. Se vor
copia valorile din vectorul device -> pe vectorul host (valus) si se va
intoarce rezultatul.

	Reshape-> se va crea un nou vector de dimensiune noua, si se
vor initializa toate cheile cu 0. Functia kernel copiaza datele de tip
asocieri (cheie, valoare) din vectorul initial, in noul vector de dimensiune
mai mare. De asemenea, pointer-ul hashmap va pointa catre noul hashmap.
	

## Cum se stochează hashtable în memoria GPU VRAM ?

	Am stocat hashtable intr-un vector de asocieri (cheie, valoare). Am
retinut in variabila hashSize dimensiunea hashmap-ului si in variabila
pairs numarul de perechi nevide (folosita fiind pentru load factor).

## Output la performanțele obținute și discutie rezultate

	In urma rularii se vor obtine urmatoarele rezultate:

------- Test T1 START	----------

HASH_BATCH_INSERT   count: 1000000          speed: 106M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 1000000          speed: 144M/sec         loadfactor: 80%
----------------------------------------------
AVG_INSERT: 106 M/sec,  AVG_GET: 144 M/sec,     MIN_SPEED_REQ: 10 M/sec

------- Test T1 END	---------- 	 [ OK RESULT: +20 pts ]



------- Test T2 START	----------

HASH_BATCH_INSERT   count: 500000           speed: 101M/sec         loadfactor: 80%
HASH_BATCH_INSERT   count: 500000           speed: 76M/sec          loadfactor: 80%
HASH_BATCH_GET      count: 500000           speed: 121M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 500000           speed: 110M/sec         loadfactor: 80%
----------------------------------------------
AVG_INSERT: 88 M/sec,   AVG_GET: 116 M/sec,     MIN_SPEED_REQ: 20 M/sec

------- Test T2 END	---------- 	 [ OK RESULT: +20 pts ]



------- Test T3 START	----------

HASH_BATCH_INSERT   count: 125000           speed: 73M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 63M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 51M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 42M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 33M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 11M/sec          loadfactor: 96%
HASH_BATCH_INSERT   count: 125000           speed: 29M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 125000           speed: 19M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 93M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 93M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 94M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 94M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 93M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 94M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 76M/sec          loadfactor: 91%
HASH_BATCH_GET      count: 125000           speed: 42M/sec          loadfactor: 91%
----------------------------------------------
AVG_INSERT: 40 M/sec,   AVG_GET: 85 M/sec,      MIN_SPEED_REQ: 40 M/sec

------- Test T3 END	---------- 	 [ OK RESULT: +15 pts ]



------- Test T4 START	----------

HASH_BATCH_INSERT   count: 2500000          speed: 114M/sec         loadfactor: 80%
HASH_BATCH_INSERT   count: 2500000          speed: 80M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 2500000          speed: 63M/sec          loadfactor: 80%
HASH_BATCH_INSERT   count: 2500000          speed: 52M/sec          loadfactor: 80%
HASH_BATCH_GET      count: 2500000          speed: 166M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 2500000          speed: 175M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 2500000          speed: 174M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 2500000          speed: 143M/sec         loadfactor: 80%
----------------------------------------------
AVG_INSERT: 77 M/sec,   AVG_GET: 164 M/sec,     MIN_SPEED_REQ: 50 M/sec

------- Test T4 END	---------- 	 [ OK RESULT: +15 pts ]



------- Test T5 START	----------

HASH_BATCH_INSERT   count: 20000000         speed: 104M/sec         loadfactor: 80%
HASH_BATCH_INSERT   count: 20000000         speed: 72M/sec          loadfactor: 80%
HASH_BATCH_GET      count: 20000000         speed: 145M/sec         loadfactor: 80%
HASH_BATCH_GET      count: 20000000         speed: 99M/sec          loadfactor: 80%
----------------------------------------------
AVG_INSERT: 88 M/sec,   AVG_GET: 122 M/sec,     MIN_SPEED_REQ: 50 M/sec

------- Test T5 END	---------- 	 [ OK RESULT: +15 pts ]

TOTAL gpu_hashtable  85/85

!! IMPORTANT !!

	Pentru obtinerea rezultatului, am dat make si rulat bench.py pe coada
dar ea ba se bloca, ba dadea teste failed (qsub si qlogin de asemenea) ceea ce a
facut rularea un calvar. 

	In mod normal, pentru obtinerea rezultatului, folosesc run_on_q.sh care,
intra pe coada hp-sl.q, da load la module si apoi ruleaza make si scriptul bench.py.

	qsub -cwd -q hp-sl.q -b y ./run_on_q.sh

	Am adaugat in arhiva fisierele .o si .e rezultate din rularea pe coada.
	
	Am observat ca daca folosesc metodele de alocare invatate la laborator,
programul functioneaza fara probleme, testele trec, dar nu intra in timp pe coada.

## Observatii

	Se poate observa ca implementarea oferita, este
mai buna decat o implementare secventiala. Acesta dureaza mai putin
datorita paralelizarii operatiilor efectuate asupra tabelei de hash.

	Managementul memoriei ocupa mare parte din timpul implementarii
paralele. De asemenea, se observa ca in toate cazurile loadFactorMax
este mai mic decat hashLoadFactor.
