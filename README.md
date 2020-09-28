Rădescu Ioana

# CUDA HashTable
	
	Acest proiect urmareste implementarea unei structuri de date tip hashtable folosind CUDA, 
	avand ca target GPU Tesla K40.

# Implementare

	Pentru crearea unei intrari din HashTable am definit urmatoarea structura: 
		typedef struct {
			unsigned int key;
			unsigned int value;
		} hashCell.
	Aceasta salveaza asocierile dintre chei si valori. Astfel, pentru definirea
	HashTable-ului se creeaza un vector de structuri.

		Pentru a evita coliziunile se utilizeaza linear probing. Prin urmare, la inserarea
	in tabela, se calculeaza hash-ul fiecarei chei, obtinandu-se un index in vector.
	De la aceasta pozitie se parcurge vectorul pana la final, cautandu-se un slot liber sau o
	pozitie in care este salvata cheia ce urmeaza sa fie inserata. Daca se ajunge la finalul
	tabelei si inserarea nu s-a putut realiza, se va cauta o pozitie care preceda indexul
	dat de hash-ul cheii. Pentru verificarea cheii de pe o anumita pozitie de utilizeaza functia
	atomicCAS().
		De asemenea, inainte de inserarea unor chei, se verifica daca tabela are destule pozitii
	libere pentru a le putea salva. In caz contrar, se apeleaza functia de reshape. Aceasta creeaza
	o tabela cu noua dimensiune si reinsereaza cheile din vechea tabela la alte poziti, utilizand
	aceeasi strategie descrisa mai sus. Aceasta functie are si rolul de a mentine un load factor
	mai mare sau egal cu 80%.
		Pentru a obtine valorile asociate unor anumite chei in tabela de hash se realizeaza o
	parcurgere similara.

# Cum se stochează hashtable în memoria GPU VRAM?

		Pentru alocarea vectorului de structuri ce formeaza HashTable-ul s-a folosit functia
	cudaMalloc().
	Astfel, in memoria GPU VRAM se vor stoca datele din tabela, in timp ce in RAM va fi salvata
	doar adresa de inceput a acestui vector.

# Output la performanțele obținute
	-------------- Test T1 --------------
	OK       +10 pts         HASH_BATCH_INSERT, 1000000, inf, 80
	OK       +10 pts         HASH_BATCH_GET, 1000000, 100, 80.0051
	TOTAL    +20 pts

	-------------- Test T2 --------------
	OK       +5 pts  HASH_BATCH_INSERT, 2000000, 66.6667, 80
	OK       +5 pts  HASH_BATCH_GET, 2000000, 200, 80.0026
	TOTAL    +10 pts

	-------------- Test T3 --------------
	OK       +5 pts  HASH_BATCH_INSERT, 2000000, 100, 80
	OK       +5 pts  HASH_BATCH_INSERT, 2000000, 100, 80
	OK       +5 pts  HASH_BATCH_GET, 2000000, 100, 80.0013
	OK       +5 pts  HASH_BATCH_GET, 2000000, 100, 80.0013
	TOTAL    +20 pts

	-------------- Test T4 --------------
	OK       +5 pts  HASH_BATCH_INSERT, 2500000, 250, 80
	OK       +5 pts  HASH_BATCH_INSERT, 2500000, 62.5, 80
	OK       +5 pts  HASH_BATCH_INSERT, 2500000, 50, 80
	OK       +5 pts  HASH_BATCH_INSERT, 2500000, 50, 80
	OK       +5 pts  HASH_BATCH_GET, 2500000, 125, 80.0005
	OK       +5 pts  HASH_BATCH_GET, 2500000, 250, 80.0005
	OK       +5 pts  HASH_BATCH_GET, 2500000, 125, 80.0005
	OK       +5 pts  HASH_BATCH_GET, 2500000, 125, 80.0005
	TOTAL    +40 pts

	TOTAL gpu_hashtable  90/90

# Concluzii:

		In cadrul acestui program se utilizeaza thread-uri CUDA pentru inserarea in HashTable 
		si pentru cautarea valorilor asociate unor chei. Acest lucru duce la o crestere semnificativa
		a performatei obtinute.
		Din output-ul obtinut se observa ca performanta scade in momentul in care este necesara 	
		realizarea unei operatii de reshape. 
		Totodata, aceasta depinde si numarul de iteratii necesare la inserare sau cautare.
