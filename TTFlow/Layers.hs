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

module TTFlow.Layers where

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude
import GHC.TypeLits
import Text.PrettyPrint.Compact (text)
import Data.Proxy

import TTFlow.TF
import TTFlow.Types

----------------
-- Helpers
matvecmulBatch :: ∀ s cols rows t. (KnownNat cols, KnownShape s) =>  Tensor (cols ': rows ': s) t -> Tensor (cols ': s) t -> Tensor (rows ': s) t
matvecmulBatch m v = squeeze0 (matmul m (expandDim0 v))

matvecmul :: Tensor (cols ': rows ': '[]) t -> Tensor (cols ': batchSize ': '[]) t -> Tensor (rows ': batchSize ': '[]) t
matvecmul m v = matmul v (transpose m)

(∙) :: Tensor '[cols, rows] t -> Tensor '[cols,batchSize] t -> Tensor '[rows,batchSize] t
(∙) = matvecmul


---------------------
-- Linear functions


-- A linear function form a to b is a matrix and a bias.
type (a ⊸ b) = (Tensor '[a,b] Float32, Tensor '[b] Float32)

-- | Apply a linear function
(#) :: (a ⊸ b) -> T '[a,batchSize] Float32 -> Tensor '[b,batchSize] Float32
(weightMatrix, bias) # v = weightMatrix ∙ v + bias

-----------------------
-- Feed-forward layers

-- | embedding layer
embedding :: ∀ embeddingSize numObjects batchSize t.
             Tensor '[numObjects, embeddingSize] t -> Tensor '[1,batchSize] Int32 -> Tensor '[embeddingSize,batchSize] t
embedding param input = gather @ '[embeddingSize] (transpose param) (squeeze0 input)


glorotUniform :: forall a b t. (KnownNat a, KnownNat b, KnownTyp t) => Tensor '[a,b] t
glorotUniform = randomUniform low high
  where
    low, high, fan_in, fan_out :: Float
    low = -4.0 Prelude.* Prelude.sqrt(6.0/(fan_in Prelude.+ fan_out)) -- use 4 for sigmoid, 1 for tanh activation 
    high = 4.0 Prelude.* Prelude.sqrt(6.0/(fan_in Prelude.+ fan_out))
    fan_in = fromIntegral (natVal (Proxy @ a))
    fan_out = fromIntegral (natVal (Proxy @ b))

denseInitialiser :: (KnownNat n, KnownNat m) => (n ⊸ m)
denseInitialiser = (glorotUniform,truncatedNormal 0.1)

dense :: ∀m n batchSize. (n ⊸ m) -> Tensor '[n, batchSize] Float32 -> (Tensor '[m, batchSize] Float32)
dense lf t = (lf # t)


------------------------
-- Convolutional layers
convInitialiser :: (KnownShape s1, KnownShape s2) =>
                   (T s1 ('Typ 'Float w), T s2 ('Typ 'Float w))
convInitialiser = (truncatedNormal 0.1, constant 0.1)

conv :: forall outChannels filterSpatialShape inChannels s t.
                  ((1 + Length filterSpatialShape) ~ Length s,
                   KnownLen filterSpatialShape,
                   KnownShape s) => -- the last dim of s is the batch size
                  (T ('[outChannels,inChannels] ++ filterSpatialShape) t, T ('[outChannels] ++ Init s) t) ->
                  T ('[inChannels] ++ s) t -> (T ('[outChannels] ++ s) t)
conv (filters,bias) input = (initLast @s (add @'[Last s] c  bias))
 where c = (convolution input filters)


maxPool2D :: forall stridex (stridey::Nat) batch height width channels.
             (KnownNat stridex, KnownNat stridey) =>
             T '[channels,width*stridex,height*stridex,batch] Float32 -> T '[channels,width,height,batch] Float32
maxPool2D (T value) = T (funcall "tf.nn.max_pool" [value
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,named "padding" (str "SAME") ])

-------------------------------
-- RNN layers and combinators

type RnnCell state input output = (state , input) -> Gen (state , output)


-- | Any pure function (feed-forward layer) can be transformed into a
-- cell by ignoring the RNN state.
-- timeDistribute :: (Tensor (a ': batchShape) -> Tensor (b ': batchShape)) -> RnnCell () (a ': batchShape) (b ': batchShape)
-- timeDistribute pureLayer (s,a) = return (s, pureLayer a)

timeDistribute :: (Tensor a t -> Tensor b t') -> RnnCell () (T a t) (T b t')
timeDistribute pureLayer (s,a) = return (s, pureLayer a)

lstm :: ∀ n x bs. (KnownNat bs) => 
        (((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n)) ->
        RnnCell (T '[n,bs] Float32, T '[n,bs] Float32) (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)
lstm (wf,wi,wc,wo) ((ht1 , ct1) , input) = do
  hx <- assign (concat0 ht1 input)
  let f = sigmoid (wf # hx)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
  c <- assign ((f ⊙ ct1) + (i ⊙ cTilda))
  h <- assign (o + tanh c)
  return ((c , h) , h)

-- | Stack two RNN cells
stackRnnLayers :: RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0,s1) a c
stackRnnLayers l1 l2 ((s0,s1),x) = do
  (s0',y) <- l1 (s0,x)
  (s1',z) <- l2 (s1,y)
  return ((s0',s1'),z)

infixr .--.
(.--.) :: forall s0 a b s1 c. RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0, s1) a c
(.--.) = stackRnnLayers

-- | @addAttention attn l@ adds the attention function @attn@ to the
-- layer @l@.  Note that @attn@ can depend in particular on a constant
-- external value @h@ which is the complete input to pay attention to.
-- The type parameter @x@ is the size of the portion of @h@ that the
-- layer @l@ will observe.
addAttention :: KnownShape s => (state -> T (x ': s) t) -> RnnCell state (T ((a+x) ': s) t) (T (b ': s) t) -> RnnCell state (T (a ': s) t) (T (b ': s) t)
addAttention attn l (s,a) = l (s,concat0 a (attn s))


-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) ->
       (state , Tensor (n ': input) t) -> Gen (state , Tensor (n ': output) u)
rnn cell (s0, t) = do
  xs <- unstack t
  (sFin,us) <- chain cell (s0,xs)
  return (sFin,stack us)

-- TODO: attempt to do this with
-- tf.foldl

-- | RNN helper
chain :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chain _ (s0 , V []) = return (s0 , V [])
chain f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (sFin,V xs') <- chain f (s1 , V xs)
  return (sFin,V (x':xs'))


-- Local Variables:
-- dante-project-root: ".."
-- End:
