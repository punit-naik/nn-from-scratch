(ns nn-from-scratch.core
  (:require [nn-from-scratch.matrix-utils :refer [create-matrix perform-arithmetic-op
                                                  matrix-multiply absolute
                                                  sigmoid transpose mean]]
            [clojure.string :refer [replace]]))

(defn train
  "Train the Neural Network for the specified number of steps
  NOTE: The optional 'rest' array value is just for storing one seed value for the
        Random 2-D Matrix Generator"
  [steps error-logging & rest-args]
  (let [input [[0 0 1] [0 1 1] [1 0 1] [1 1 1]]
        expected-output [[0] [1] [1] [0]]]
    (loop [stepz (range 1 (inc steps))
           ;; Initial Synapses/Weights
           synapse-0 (-> (if (zero? (count rest-args))
                           (create-matrix 3 4)
                           (create-matrix 3 4 (first rest-args)))
                         (perform-arithmetic-op 2 *)
                         (perform-arithmetic-op 1 -))
           synapse-1 (-> (if (zero? (count rest-args))
                           (create-matrix 4 1)
                           (create-matrix 4 1 (first rest-args)))
                         (perform-arithmetic-op 2 *)
                         (perform-arithmetic-op 1 -))
           predicted-output nil]
      
        (if (empty? stepz)
          {:predicted predicted-output :expected expected-output}
          (let [;; Layers
                layer-0 input
                layer-1 (->> (matrix-multiply layer-0 synapse-0)
                            sigmoid)
                layer-2 (->> (matrix-multiply layer-1 synapse-1)
                            sigmoid)
                ;; Back Propagation
                layer-2-error (perform-arithmetic-op expected-output layer-2 -)
                _ (when (and error-logging
                            (zero? (mod (first stepz) (/ steps (Integer/parseInt (replace (str steps) #"0" ""))))))
                    (println "Error :" (->> (absolute layer-2-error) mean)))
                ;; Deltas
                layer-2-delta (-> layer-2-error
                                  (perform-arithmetic-op
                                  (sigmoid layer-2 true) *))
                layer-1-error (->> (transpose synapse-1)
                                  (matrix-multiply layer-2-delta))
                layer-1-delta (-> layer-1-error
                                  (perform-arithmetic-op
                                  (sigmoid layer-1 true) *))]
            (recur
              (rest stepz)
              ;; Update Synapses (Weights)
              (perform-arithmetic-op synapse-0
                                     (matrix-multiply
                                      (transpose layer-0)
                                      layer-1-delta) +)
              (perform-arithmetic-op synapse-1
                                     (matrix-multiply
                                      (transpose layer-1)
                                      layer-2-delta) +)
              ;; Update result
              layer-2))))))

(defn train-and-print
  "Trains the Neural Network and Print the result on the console"
  [steps error-logging]
  (let [{:keys [expected predicted]} (train steps error-logging)]
    (println "Actual Output    :" expected)
    (println "Predicted Output :" predicted)))

(defn -main
  [& args]
  (if (zero? (count args))
    (train-and-print 60000 true)
    (train-and-print (Integer/parseInt (first args)) true)))