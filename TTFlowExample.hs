{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}

import TTFlow

example1 :: KnownNat batchSize => Tensor '[20,batchSize] 'Int32 -> Gen (Tensor '[20,batchSize] 'Float32)
example1 input' = do
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
main = writeFile "ttflow_example.py" (generate $ compile @1024 example1 crossEntropy)



