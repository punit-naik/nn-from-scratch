(ns nn-from-scratch.core-test
  (:require [clojure.test :refer [deftest are testing]]
            [nn-from-scratch.core :refer [train]]))

(deftest nn-purity
  (testing
   "Checking the purity of the NN functions"
   (let [[[e-1] [e-2] [e-3] [e-4]] (:predicted (train 60000 false 1000))
         [[p-1] [p-2] [p-3] [p-4]] (:predicted (train 60000 false 1000))]
     (are [x-1 x-2] (= x-1 x-2)
          e-1 p-1
          e-2 p-2
          e-3 p-3
          e-4 p-4))))

(deftest nn-precision
  (testing
   "Checking error between actual and predicted output values"
   (let [[[e-1] [e-2] [e-3] [e-4]] [[0] [1] [1] [0]]
         [[p-1] [p-2] [p-3] [p-4]] (:predicted (train 60000 false))]
     (are [x-1 x-2 y] (< (Math/abs (- x-1 x-2)) y)
       e-1 p-1 0.006
       e-2 p-2 0.006
       e-3 p-3 0.006
       e-4 p-4 0.006))))
