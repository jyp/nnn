{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
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

module TTFlow.Learn where

import TTFlow.Types
import TTFlow.TF
import qualified Prelude ()
import Prelude (($),(=<<),return)
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last)


-------------------
-- Loss functions

type Loss s bs t = Last s ~ bs => Tensor s t -> Tensor s t -> Tensor '[bs] 'Float32


crossEntropy :: Tensor '[n,bs] 'Float32 -> Tensor '[n,bs] 'Float32 -> Tensor '[bs] 'Float32
crossEntropy y_ y = negate (reduceSum0 (y_ ⊙ log y))

softmaxCrossEntropyWithLogits :: Tensor '[numClasses,batchSize] 'Float32 -> Tensor '[numClasses,batchSize] 'Float32 -> Tensor '[batchSize] 'Float32
softmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

--------------------------------
-- Model maker.

type Batch s batchSize = Tensor (s++'[batchSize])

compile :: ∀ bs s (s'::Shape) t u. (Last s'~bs, KnownShape s, KnownShape s',KnownTyp t, KnownTyp u) =>
          (Tensor s t -> Gen (Tensor s' u)) -> Loss s' bs u -> Gen ()
compile model lossFunction = do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [] $ do
    x <- placeholder "x"
    y_ <- placeholder "y_"
    y <- assign =<< model x
    loss <- assign (reduceMeanAll (lossFunction y y_))
    gen (text "return " <> tuple [fromTensor x,fromTensor y,fromTensor y_,fromTensor loss])
    return ()



-- Local Variables:
-- dante-project-root: ".."
-- End:
