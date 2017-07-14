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

atShape :: forall s t. T s t -> T s t
atShape x = x

mnist :: forall batchSize. KnownNat batchSize => Model [784,batchSize] Float32  '[10,batchSize] Float32
mnist input gold = do
  filters1 <- parameter "f1" convInitialiser
  filters2 <- parameter "f2" convInitialiser
  w1 <- parameter "w1" denseInitialiser
  w2 <- parameter "w2" denseInitialiser
  let nn = arrange3                           #>
           atShape @'[1,28,28,batchSize]      #>
           (relu . conv @32 @'[5,5] filters1) #>
           atShape @'[32,28,28,batchSize]     #>
           maxPool2D @2 @2                    #>
           atShape @'[32,14,14,batchSize]     #>
           (relu . conv @64 @'[5,5] filters2) #>
           maxPool2D @2 @2                    #>
           linearize3                         #>
           (relu . dense @1024 w1)            #>
           dense @10 w2
      logits = nn input
  categoricalDistribution logits gold

-- agreement :: KnownNat batchSize => Tensor '[20,batchSize] Int32 -> Gen (Tensor '[20,batchSize] Float32)
-- agreement input' = do
--   let input = expandDim1 input'
--   (embs,lstm1,lstm2,w) <- parameter "params"
--   (_sFi,out) <- rnn (timeDistribute (embedding @50 @100000 embs)
--                      .--.
--                      (lstm @150 lstm1)
--                      .--.
--                      (lstm @150 lstm2)
--                      .--.
--                      timeDistribute (sigmoid . squeeze0 . dense  w))
--                 (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
--   return out

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


main :: IO ()
main = writeFile "ttflow_example.py" (generate $ compile (mnist @None)) >> putStrLn "zog zog"

{-> main

*** Exception: ghc: signal: 15
-}


