March 24, 2026 (en medio de la Collaboration Meeting)

Mi idea es comparar las single-muon probabilites, que se calcularon previamente y se encuentra en:

/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/single_muon_prob/pmt/outputs/csv/ (PARA PMT)

/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/single_muon_prob/sipm/outputs/csv/ (PARA SIPM)

Lo que me genera curiosidad es que los p_1mu_neigh y p_1mu_non_neigh de PMT y SiPM son parecidos. A priori, como los PMT
son mas ineficientes, hubiera esperado que p_1mu_neigh (ie corner-clipping) sea menor.
Tambien, por cómo se ven los histogramas dt de crosstalk, esperaria que los PMT tuvieran mas p_1mu_non_neigh (ie crosstalk).

Sospecho que esto puede tener que ver con la normalizacion que uno hace:

p_mu_neigh = n1mu_neigh / (n1mu_neigh + n1mu_non_neigh + n1bar)

donde se asume que

i) n0 (ie, muones ineficientes que no activan ninguna barra) = 0 -> no es lo mas adecuado para PMT
ii) n1bar = n(1bar | 1mu)

Respecto a ii), estamos asumiendo que todos los eventos que tienen una barra con pattern corresponden a solo 1 muon.
Formalmente, sabemos que n1bar = n(1bar|1mu) + n(1bar | 2mu) + n(1bar | 3mu) + ...
Es decir, hay eventos de 2 muones en el que solo 1 activa una barra; 3 muones que solo 1 activa una barra, etc ...

Para SiPM, podemos asumir de manera masomenos segura que esos terminos de orden superior son despreciables, porque
SiPM son mas eficientes. Sin embargo, para PMT ese podría no ser el caso, y entonces se estaría sobre-estimando
el denominador al asumir n1bar = n(1bar | 1mu)
