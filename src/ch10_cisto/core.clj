(ns ch10-cisto.core
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release]]            
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer :all]
             [legacy :refer :all]
             [constants :refer :all]
             [toolbox :refer :all]
             [utils :refer :all]]
            [vertigo
             [bytes :refer [buffer direct-buffer byte-seq byte-count slice]]
             [structs :refer [int8 int32 int64 wrap-byte-seq]]])
  (:import [org.jocl CL]))


(set! *unchecked-math* true)
;(try 
 (with-release [;dev (nth  (sort-by-cl-version (devices (first (platforms)))) 0)
                platformsone (first (platforms))
                 
                dev (nth  (sort-by-cl-version (devices platformsone)) 0)
                ;dev (nth  (sort-by-cl-version (devices platformsone)) 1)
                ctx (context [dev])
                 cqueue (command-queue-1 ctx dev :profiling)
                ]
  
 (println (vendor platformsone))
 (println "dev: " (name-info dev))  
   

  (facts
   "Listing on page 225."
   (let [program-source
         (slurp (io/reader "examples/reduction.cl"))
         num-items (Math/pow 2 20)                    ;2 na 20-tu = 1048576
         bytesize (* num-items Float/BYTES)           ;Float/BYTES = 4    =>   bytesize = 4 * 2na20 = 4 * 1048576 = 4194304.0
         workgroup-size 256
         notifications (chan)
         follow (register notifications)
         
         data (float-array (repeatedly num-items #(rand-int num-items)))
         ;data (float-array (repeatedly num-items #(let [p (rand-int num-items)]
         ;                                           (println p)
         ;                                           p)))
         
         cl-partial-sums (* workgroup-size Float/BYTES)       ;4 * 256 = 1024
         partial-output (float-array (/ bytesize workgroup-size))      ;4*2na20 / 256 = 4*2na20 / 2na8 = 4*2na12   - niz od 16384 elemenata
         output (float-array 1)               ;pocetna vrednost jedan clan sa vrednoscu 0.0
         ]   
     (with-release [cl-data (cl-buffer ctx bytesize :read-only)
                    cl-output (cl-buffer ctx bytesize :write-only)
                    cl-partial-output (cl-buffer ctx (/ bytesize workgroup-size)   ;kreira cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
                                                 :read-write)
                    prog (build-program! (program-with-source ctx [program-source]))   ;kreira program u kontekstu ctx sa kodom programa u kojem se nalaze tri kernela 
                    naive-reduction (kernel prog "naive_reduction")            ;definise kernel iz prog
                    reduction-scalar (kernel prog "reduction_scalar")          ;definise kernel iz prog
                    reduction-vector (kernel prog "reduction_vector")          ;definise kernel iz prog
                    reduction-complete (kernel prog "reduction_complete")      ;definise kernel iz prog                    
                    profile-event (event)                  ;kreira novi cl_event (dogadjaj)
                    profile-event1 (event)                 ;          -||-
                    profile-event2 (event)                 ;          -||- 
                    profile-event3 (event)]                ;          -||-       
       ;(println "(apply + (float-array (range 0" num-items "))): " (apply + data))

       (facts
         
       (println "============ Naive reduction ======================================")
       
        ;; ============ Naive reduction ======================================
        (set-args! naive-reduction cl-data cl-output) => naive-reduction
        (enq-write! cqueue cl-data data) => cqueue                                 ;SETUJE VREDNOST GLOBALNE PROMENJIVE cl-data SA VREDNOSCU data
        
        ;(println "data: " (seq data))        
        (enq-nd! cqueue naive-reduction (work-size [1]) nil profile-event)
        => cqueue
        (follow profile-event) => notifications
        (enq-read! cqueue cl-output output) => cqueue
        (finish! cqueue) => cqueue
        (println "Naive reduction time:"
                 (-> (<!! notifications) :event profiling-info durations :end))
        (println "Naive output: " (seq output))
        (println "sta je data: " data)        
        ;(aget output 0) => num-items
        (println "============ Scalar reduction ======================================")
        ;; ============= Scalar reduction ====================================
         (set-args! reduction-scalar cl-data cl-partial-sums cl-partial-output)  ;setovanje promenjivih u kernelu        reduction-scalar
                                                                                 ;cl-partial-sums=1024  i         
        => reduction-scalar                                                      ;cl-partial-output = cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
        (enq-nd! cqueue reduction-scalar                       ;asinhrono izvrsava kernel u uredjaju. cqueue, kernel koji se izvrsava
                 (work-size [num-items] [workgroup-size])      ;[2na20] [256] 
                 nil profile-event)                            ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
        (follow profile-event)
        (enq-read! cqueue cl-partial-output partial-output)
 
        ;(long (first partial-output)) => workgroup-size
        ;(println "partial-output POJEDINACNA RESENJA: " (seq partial-output))

       
        (finish! cqueue)
        (println "Scalar reduction time:"
                 (-> (<!! notifications) :event profiling-info durations :end))
       (println "UKUPAN ZBIR MEDJUSUMA" (apply + (float-array (seq partial-output))))
        
       (println "============ Vector reduction ======================================")
        ;; =============== Vector reduction ==================================
         (set-args! reduction-vector cl-data cl-partial-sums cl-partial-output)       ;setovanje polja u kernelu
        => reduction-vector
        (enq-nd! cqueue reduction-vector                           ;asinhrono izvrsava kernel(kernel) u uredjaju(dev) sa listom kernela(queue)    queue kernel
         (work-size [(/ num-items 4)] [workgroup-size])    ;work size - [broj elelmenata 2na20 / 4 =[65536] [256]
         nil profile-event1)                               ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
        (follow profile-event1)   ;postavlja event1
        
        ;(enq-read! cqueue cl-partial-output partial-output)

        
        (finish! cqueue) 
        (println "Vector reduction time:" 
                 (-> (<!! notifications) :event profiling-info durations :end))
        (println "VEKTOR ZBIR MEDJUSUMA" (apply + (float-array (seq partial-output))))        
        ;(first partial-output) => num-items
        ;(println "output: " (seq output))
        
        (println "============ Complete reduction ======================================")
        ;; =============== Complete reduction ==================================
         (set-args! reduction-complete cl-data cl-partial-sums cl-partial-output)       ;setovanje polja u kernelu
        => reduction-complete
        (enq-nd! cqueue reduction-complete                 ;asinhrono izvrsava kernel(kernel) u uredjaju(dev) sa listom kernela(queue)    queue kernel
         (work-size [(/ num-items 4)] [workgroup-size])    ;work size - [broj elelmenata 2na20 / 4 =[65536] [256]
         nil profile-event2)                               ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
        (follow profile-event2)   ;postavlja event1
        
        ;rezultat iz prethodnog kernela stavljamo kao ulaz u isti taj kernel
        (set-args! reduction-complete cl-partial-output cl-partial-sums cl-partial-output)  ;setovanje promenjivih u kernelu   reduction-vector
                                                                                          ;cl-partial-sums=1024  i                                                                              
        => reduction-complete                                                               ;cl-partial-output = cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
        (enq-nd! cqueue reduction-complete                                            ;asinhrono izvrsava kernel u uredjaju. cqueue, kernel koji se izvrsava
                 (work-size [(/ num-items 4 workgroup-size 4)] [workgroup-size])    ;2na20 / 4 / 256 / 4 = [256] [256]
                 nil profile-event3)                                                ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
        (follow profile-event3)   ;postavlja event3
        
        (enq-read! cqueue cl-partial-output partial-output)
      
 
        (finish! cqueue) 
        (println "Complete reduction time:" 
                 (-> (<!! notifications) :event profiling-info durations :end)
                 (-> (<!! notifications) :event profiling-info durations :end))
        (println "VEKTOR ZBIR MEDJUSUMA 2222222" (first (seq partial-output)))          
        ;(first partial-output) => num-items
        ;(println "output: " (seq output))        


        
        ;(println "============ Ostalo ======================================")           
        (println "num-items: " num-items) 
        ;(println "output: " (seq output))
        (println "---------------KRAJ -------------------")        
        )))))
  
 ;(catch Exception e (println "Greska 11111111: " (.getMessage e))))
