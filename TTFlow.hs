{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude
import Text.PrettyPrint.Compact hiding (Last)
import Data.List (intercalate)
import GHC.TypeLits
import Data.Proxy
import Data.Monoid hiding (Last)
import Control.Monad (ap)

import TTFlow.TF
import TTFlow.Types
import TTFlow.Layers


example1 :: KnownNat batchSize => Tensor '[20,1,batchSize] 'Int32 -> Gen (Tensor '[20,batchSize] 'Float32)
example1 input = do
  (embs,lstm1,lstm2,w) <- parameter "params"
  (_sFi,out) <- rnn (timeDistribute (embedding @ 50 @ 100000 embs)
                     .--.
                     (lstm @ 150 lstm1)
                     .--.
                     (lstm @ 150 lstm2)
                     .--.
                     timeDistribute (sigmoid . squeeze0 . dense  w))
                (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  return out

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>

type Batch s batchSize = Tensor (s++'[batchSize])


compile :: ∀ bs s (s'::Shape) t u. (Last s'~bs, KnownShape s, KnownShape s',KnownTyp t, KnownTyp u) =>
          (Tensor s t -> Gen (Tensor s' u)) -> Loss s' bs u -> Gen ()
compile model lossFunction = do
  x <- placeholder "x"
  y_ <- placeholder "y_"
  y <- assign =<< model x
  "y" <-- y
  "loss" <-- reduceMeanAll (lossFunction y y_)
  return ()


generate :: Gen () -> String
generate s = unlines (fromGen s (\() _v0 -> []) 0)

main :: IO ()
main = putStrLn $ generate $ compile @1024 example1 crossEntropy



