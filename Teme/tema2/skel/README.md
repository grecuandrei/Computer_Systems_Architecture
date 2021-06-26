# Tema 2 - Optimizarea inmultirilor de matrice

# Autor: Grecu Andrei-George

**Conventie:** Pentru usurinta am folosit simbolul de matrice transpusa `'`

Se cere implementarea operatiei `A * B * B' + A' * A` cu diverse optimizari si
se compara performantele.

Pentru a se masura mai bine performantele, am pus un input extins `input_extended`
care testeaza pentru 13 inputuri.

## Optimizari ale algoritmului de inmultire
Matricea `A` fiind superior triunghiulara, indicii au valori modificate dupa care
se va parcurge aceasta matrice, pentru a renunta la operatii inutile
(inmultiri cu `0`).

### C = A * B * B' + A' * A

Pentru variantele `blas` si `opt_m` am comasat cate o inmultire de matrice.
Respectiv, pentru `blas` am utilizat functia `dgemm` pentru inmultirea rezultatului
anterior obtinut (`C = A * B`) cu `B'` si adunarea cu rezultatul `A' * A`,
iar pentru `opt_m` in ultimele bucle am facut inmultirea `A' * A` si am adunat direct
in `C`, care avea rezultatul anterior `A * B * B'`.

## neopt
Metoda aplica optimizarea generala de mai sus, care tine cont in mare parte doar
de forma matricei `A`.

Se obtin urmatoarele performante:

```
Run=./tema2_neopt: N=400: Time=1.465122
Run=./tema2_neopt: N=500: Time=2.773250
Run=./tema2_neopt: N=600: Time=4.717827
Run=./tema2_neopt: N=700: Time=7.511213
Run=./tema2_neopt: N=800: Time=11.389843
Run=./tema2_neopt: N=900: Time=15.893691
Run=./tema2_neopt: N=1000: Time=21.378414
Run=./tema2_neopt: N=1100: Time=28.685242
Run=./tema2_neopt: N=1200: Time=39.281281
Run=./tema2_neopt: N=1300: Time=52.478832
Run=./tema2_neopt: N=1400: Time=65.634033
Run=./tema2_neopt: N=1500: Time=88.351936
Run=./tema2_neopt: N=1600: Time=110.780083
```

## opt_blas

O metoda chiar surprinzatoare (fapt intarit de catre timpii de rulare).
Implementarea mea se bazeaza pe optimizarile aduse de biblioteca _BLAS_ la
nivelul functiilor `cblas_dtrmm` (care inmulteste o matrice triunghiulara (`A` in
cazul de fata) cu un scalar (`1` in acest caz) si apoi inmulteste rezultatul cu
alta matrice (`B` sau `A`)) si `cblas_dgemm` (care inmulteste matricea `C` cu o
alta matrice transpusa `B` si apoi aduna rezultatul cu alta matrice (`A' * A`)).

Astfel, timpii obtinuti sunt:
```
Run=./tema2_blas: N=400: Time=0.059127
Run=./tema2_blas: N=500: Time=0.100501
Run=./tema2_blas: N=600: Time=0.151246
Run=./tema2_blas: N=700: Time=0.211067
Run=./tema2_blas: N=800: Time=0.262765
Run=./tema2_blas: N=900: Time=0.359151
Run=./tema2_blas: N=1000: Time=0.488576
Run=./tema2_blas: N=1100: Time=0.651596
Run=./tema2_blas: N=1200: Time=0.865874
Run=./tema2_blas: N=1300: Time=1.066964
Run=./tema2_blas: N=1400: Time=1.322046
Run=./tema2_blas: N=1500: Time=1.648984
Run=./tema2_blas: N=1600: Time=1.973090
```

## opt_m

In aceasta imlpementare s-au folosit 4 tipuri de optimizari.

### Optimizarea pentru accesul la memorie
Pentru a oferi un bonus de optimizare, s-a folosit si aceasta tehnica de
reordonare a buclelor, mai specific la primele 2 calcule unde s-a folosit
configuratia `kij`, care ar trebui sa aiba cele mai bune performante
conform _Laboratorul 5_.

### Obtinerea localitatii spatiale
In cadrul algoritmului clasic de inmultire de matrice, operantul din dreapta
este parcurs pe coloane, ceea ce nu confera algoritmului localitate spatiala,
ceea ce va rezulta intr-un numar mare de cache missuri. Solutia este sa transpunem
matricea din dreapta si sa o parcurgem pe linii.

### Eliminarea constantelor din bucle
Similar implementarilor din _Laboratorul 5_, s-a renuntat la calculul
indexilor de tipul `i * N + j` prin folosirea pointerilor, care sunt
incrementati pentru fiecare pozitie din matrice sau pentru fiecare termen din
suma, astfel scazand numarul total de operatii (nu in virgula mobila).

### Folosirea registrelor procesorului
Toate datele necesare unei inmultiri de matrice (indicii pentru pozitiile din
matrice ale operanzilor, pointerii mentionati mai sus) sunt retinute direct in
registrele procesorului (folosind tipuri de date `register <tip>`). Aceasta tehnica
se foloseste pentru a se elimina overheadurile de acces la memorie (chiar si cache)
pentru aceste variabile.

Rezultatele testelor se pot observa mai jos:
```
Run=./tema2_opt_m: N=400: Time=0.269472
Run=./tema2_opt_m: N=500: Time=0.529345
Run=./tema2_opt_m: N=600: Time=0.900567
Run=./tema2_opt_m: N=700: Time=1.428457
Run=./tema2_opt_m: N=800: Time=2.110324
Run=./tema2_opt_m: N=900: Time=2.903776
Run=./tema2_opt_m: N=1000: Time=3.891063
Run=./tema2_opt_m: N=1100: Time=5.109972
Run=./tema2_opt_m: N=1200: Time=6.490645
Run=./tema2_opt_m: N=1300: Time=9.059606
Run=./tema2_opt_m: N=1400: Time=11.738103
Run=./tema2_opt_m: N=1500: Time=14.449757
Run=./tema2_opt_m: N=1600: Time=17.616587
```

## Grafice
Graficele timpilor de executie in functie de `N` se pot observa dupa rularea
scriptului `plot_graphics.py` astfel:
```
$ python3 plot_graphics.py
```

Timpii din acest _README_ sunt preluati de catre script din fisierul
`ibm_nehalem_runtimes.json`.
Plotul este salvat in `graphic_extra.png`, iar pentru a se observa mai in detaliu
diferentele, a fost plotat si un grafic care nu include toti timpii de executie
ai lui `tema2_neopt`.

## Cachegrind

**Mentiune:** Am rulat cachegrind-ul pe fep, nu pe coada deoarece nu am reusit
sa dau submit la job, data fiind coada plina.

De departe se poate vedea performanta bibliotecii _BLAS_ care exceleaza in miss
rates si numar de referinte. Totusi se poate observa o problema cand vine vorba
de misprediction rate care este mai mare in comparatie cu celelalte implementari.

O comparatie mai elocventa este intre implementarile `neopt` si `opt_m` la care
se poate observa o diferenta notabila.
Daca ne uitam la referintele din I, D si LL se poate observa ca acestea sunt mult
mai putine la varianta optimizata decat in cazul celalalt.
Ca si miss-uri in D1 se poate spune ca aici este diferenta majora. Varianta optimizata
avand un miss rate de doar `2.9%`, iar cea neoptimizata `4.2%`.

## Concluzii

Analizand graficele create de scriptul de mai sus, precum si timpii mentionati
la fiecare exercitiu, putem face urmatoarele observatii pe baza performantelor:

Accesul la date este foarte important pentru procesor si nu numai.

Se remarca o performanta foarte buna in cazul variantei optimizate fata de cea
neoptimizata, intarind faptul ca reordonarea buclelor, eliminarea constantelor
si folosirea registrelor ajuta in implementarea mai optima a inmultirii matricilor.

Graficul metodei `tema2_blas` este aproape o dreapta =)).