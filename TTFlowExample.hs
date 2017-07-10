{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}

import TTFlow

(#>) :: forall b c a. (a -> b) -> (b -> c) -> a -> c
(#>) = flip (.)

mnist :: forall batchSize. KnownNat batchSize => Model [1,28,28,batchSize] Float32  '[batchSize] Int64
mnist input gold = do
  (filters1,filters2) <- parameter "conv"
  (w1,w2) <- parameter "dense"
  let nn = (relu . conv @32 @'[5,5] filters1) #>
           maxPool2D @2 @2 #>
           (relu . conv @64 @'[5,5] filters2) #>
           maxPool2D @2 @2 #>
           (reshape2 . reshape2) #>
           (relu . dense @1024 w1) #>
           dense @10 w2
      logits = nn input
  categorical logits gold

agreement :: KnownNat batchSize => Tensor '[20,batchSize] Int32 -> Gen (Tensor '[20,batchSize] Float32)
agreement input' = do
  let input = expandDim1 input'
  (embs,lstm1,lstm2,w) <- parameter "params"
  (_sFi,out) <- rnn (timeDistribute (embedding @50 @100000 embs)
                     .--.
                     (lstm @150 lstm1)
                     .--.
                     (lstm @150 lstm2)
                     .--.
                     timeDistribute (sigmoid . squeeze0 . dense  w))
                (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  return out

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


main :: IO ()
main = writeFile "ttflow_example.py" (generate $ compile (mnist @1024))



