__kernel void naive_reduction(__global float* data, __global float* output) {
    float sum = 0.0;
    int lid = get_local_id(0);
    if (get_global_id(0) == 0) {
        for (int i = 0; i < 1048576; i++) {
            sum += data[i];           
        }
    }
    *output = sum;
}


// Formirano je 4,096 work-group sa po 256 work-items (4,096*256=1,048,576 tj. 2na20)
// work-group velicina =256 (work-items). Definisano u clojure. (enq-nd! cqueue reduction-scalar (work-size [num-items] [workgroup-size (=256 mi smo definisali)]) nil profile-event)
__kernel void reduction_scalar(__global float* data,           //2na20 elemeneta koji su smesteni u GLOBALNU MEMORIJU pozivom kernel 0,1,2,3,4,5,6,7, ... ,1048575

                               __local float* partial_sums, //Lokalne medjusume. Npr:
                                                            //Za prvu globalnu(izlaznu - output) medjusumu postoji 16 medjusuma sa medjuzbirovima po 16 clanova 
                                                            // 16 clanova = "get_local_size(0)" 256 / 16 "banaks u jednom memorijskom bloku"
                                                            //  0+  1+  2+ ... + 15 = 120  partial_sums[0]
                                                            // 16+ 17+ 18+ ... + 31 = 376  partial_sums[1]
                                                            // 32+ 33+ 34+ ... + 47 = 632  partial_sums[2]
                                                            //     ...
                                                            //240+241+242+ ... +255= 3960  partial_sums[15].Zbir ovih 16 lokalnih medjusuma je zbir prve globalne medjusume output[0]=32,640
                                                                 
                               __global float* output) {  //Izlaz iz kernela(ima 4,096 clanova). Niz medjusuma koje su zbir po 256 polja iz globalne memorije data jer je lokalna velicina 256
                                                          //  0- 255 =  32640   0+1+2+3+4+5+ ... +254+255   = 32640  Zbir prvih 16 lokalnih medjusuma partial_sums ili 256 prvih clanova
                                                          //256- 511 =  98176   256+257+258+ ... +510+511   = 98176  Zbir drugih 16 lokalnih medjusuma partial_sums ili 256 drugih clanova
                                                          //512- 767 = 163712   512+513+514+ ... +766+767   =163712  Zbir trecih 16 lokalnih medjusuma partial_sums ili 256 trcih clanova
                                                          //768-1023 = 229248   768+769+770+ ... +1022+1023 =229248  Zbir cetvrtih 16 lokalnih medjusuma partial_sums ili 256 cetvrtih clanova 
                                                          //          ...                    ...                              ...                           

                     //Lokalna memorija je organizovana preko memorijske banke cija velicina zavisi od arhitekture uredjaja (16 ili 32 banks) u jednom memorijskom bloku
                     //kod grafike Intel je to 16 banaks u jednom memorijskom bloku
    int lid = get_local_id(0);         //Uzima po 16 vrednosti u intervalu 0-255 (na svakih 16, 0-15, 16-31, 32-47, ... , 240-255).
                                       //Lokalni ID se ponavlja ali u razlicitim work-group.Zato lid-ova ima samo 256 jer je to velicina work-item.
    int gsize = get_local_size(0);     //256. Definisano u clojure. (enq-nd! cqueue reduction-scalar (work-size [num-items] [workgroup-size (=256 mi smo definisali)]) nil profile-event)
    
    //printf("%u\n", get_local_id(0));   

    partial_sums[lid] = data[get_global_id(0)];    //Transver iz globalne memorije u lokalnu. 
                                                   //Za svaki work-group ucitava se podskup ulaznih podataka tj. 256 work-item-a. Na osnovu ovog upisa racuna se medjusuma work-group
    barrier(CLK_LOCAL_MEM_FENCE);                  //Blokira izvrsenje dok se ne prebace svi work-item iz globalne u lokalnu memoriju


    for (int i = gsize/2; i > 0; i >>= 1) {         //256/2=128 (binarno 1000000 i>>=1 dobijamo      (>>  right shift. ako je neki veci broj pomera se za toliko mesta binarni zapis)
                                                    //                   0100000 sto je 64 i>>=1     (<<  left  shift. kontra umesto u desno pomera se vrednost u levo tj. vrednost raste)
                                                    //                   0010000 sto je 32 i>>=1
                                                    //                   0001000 sto je 16 i>>=1
                                                    //                   0000100 sto je 8  i>>=1
                                                    //                   0000010 sto je 4  i>>=1
                                                    //                   0000001 sto je 2  i>>=1
                                                    //                   0000000 ali se zbog = izvrsava 1                   
                                                    //                   0000000 sto je 0 nema dalje i ne izvrsava se ova petlja )
        if (lid < i) {    
            partial_sums[lid] += partial_sums[lid + i];          //sabiraju se samo one vrednosti koje su ispod polovine niza (da bi postojale vrednosti u drugoj polovini)
        }
        barrier(CLK_LOCAL_MEM_FENCE);   //Nakon svakog kruga barrier Prinudi svaki work-item da saceka dok svaki work-item, u okviru work-group, ne pristupi lokalnoj memoriji
    }

    if(lid == 0) {
        output[get_group_id(0)] = partial_sums[0];        //Vraca vrednost pojedinacne work-group (Na kraju krajeva za svih 4,096 komada)
    }
}

__kernel void reduction_vector(__global float4* data,
                               __local float4* partial_sums,
                               __global float4* output) {

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    partial_sums[lid] = data[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = group_size/2; i>0; i >>= 1) {
        if(lid < i) {
            partial_sums[lid] += partial_sums[lid + i];
        }
          barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        //output[get_group_id(0)] = dot (partial_sums[0], (float4)(1.0f));   //ovo kada je output samo float a ne vektor float4
        output[get_group_id(0)] = partial_sums[0];        
    }
}

__kernel void reduction_complete(__global float4* data,
                                 __local float4* partial_sums,
                                 __global float* output) {

    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    partial_sums[lid] = data[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = group_size/2; i>0; i >>= 1) {
        if(lid < i) {
            partial_sums[lid] += partial_sums[lid + i];
        }
          barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //output[get_group_id(0)] = partial_sums[0].s0 + partial_sums[0].s1 +
    //                          partial_sums[0].s2 + partial_sums[0].s3; 
    
    if(lid == 0) {
        output[get_group_id(0)] = dot (partial_sums[0], (float4)(1.0f));
    }
}