(ns nn-from-scratch.core
  (:require [nn-from-scratch.matrix-utils :as matrix-utils]
            [clojure.string :refer [replace]]))

(defn train
  "Train the Neural Network for the specified number of steps"
  [steps]
  (let [input [[0 0 1] [0 1 1] [1 0 1] [1 1 1]]
        expected-output [[0] [1] [1] [0]]
        predicted-output (atom [])
        synapse-0 (atom
                    (-> (matrix-utils/create-matrix 3 4)
                        (matrix-utils/perform-arithmetic-op 2 *)
                        (matrix-utils/perform-arithmetic-op 1 -)))
        synapse-1 (atom
                    (-> (matrix-utils/create-matrix 4 1)
                        (matrix-utils/perform-arithmetic-op 2 *)
                        (matrix-utils/perform-arithmetic-op 1 -)))]
    (doseq [i (range 1 (inc steps))]
      (let [current-synapse-0 @synapse-0
            current-synapse-1 @synapse-1
            ; Layers
            layer-0 input
            layer-1 (->> (matrix-utils/matrix-multiply layer-0 current-synapse-0)
                         matrix-utils/sigmoid)
            layer-2 (->> (matrix-utils/matrix-multiply layer-1 current-synapse-1)
                         matrix-utils/sigmoid)
            ; Back Propagation
            layer-2-error (matrix-utils/perform-arithmetic-op expected-output layer-2 -)
            _ (if (zero? (mod i (/ steps (Integer/parseInt (replace (str steps) #"0" "")))))
                (println "Error :" (->> (matrix-utils/absolute layer-2-error) matrix-utils/mean)))
            ; Deltas
            layer-2-delta (-> layer-2-error
                              (matrix-utils/perform-arithmetic-op
                                (matrix-utils/sigmoid layer-2 true) *))
            layer-1-error (->> (matrix-utils/transpose current-synapse-1)
                               (matrix-utils/matrix-multiply layer-2-delta))
            layer-1-delta (-> layer-1-error
                              (matrix-utils/perform-arithmetic-op
                                (matrix-utils/sigmoid layer-1 true) *))]
        ; Update Synapses (Weights)
        (reset! synapse-1 (matrix-utils/perform-arithmetic-op
                            current-synapse-1
                            (matrix-utils/matrix-multiply
                              (matrix-utils/transpose layer-1)
                              layer-2-delta) +))
        (reset! synapse-0 (matrix-utils/perform-arithmetic-op
                            current-synapse-0
                            (matrix-utils/matrix-multiply
                              (matrix-utils/transpose layer-0)
                              layer-1-delta) +))
        ; Store the result in a global variable
        (reset! predicted-output layer-2)))
    @predicted-output))

(defn train-and-print
  "Trains the Neural Network and Print the result on the console"
  [steps]
  (println "Predicted Output :" (train steps)))

(defn -main
  [& args]
  (if (zero? (count args))
    (train-and-print 60000)
    (train-and-print (Integer/parseInt (first args)))))