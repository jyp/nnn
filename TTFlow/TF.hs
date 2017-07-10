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

module TTFlow.TF where

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude
import Text.PrettyPrint.Compact hiding (Last)
import GHC.TypeLits
import Data.Proxy
import TTFlow.Types

zeros :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
zeros = T (funcall "tf.zeros" [(showShape @ shape)])

ones :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
ones = T (funcall "tf.ones" [(showShape @ shape)])

-- | Declare a parameter to optimize.
parameter' :: ∀ (shape :: Shape) t. String -> T shape t -> Gen (T shape t)
parameter' name (T initial) = do
  v <- newVar
  v <-- T (funcall "tf.Variable" [initial, text "name=" <> string (show (name))])
  return (T v)

placeholder :: ∀t s. (KnownShape s, KnownTyp t) => String -> Gen (T s t)
placeholder n = do
  let name = text n
  name <-- T (funcall "tf.placeholder" [showTyp @t, text "shape=" <> showShape @ s])
  return (T name)

reduceAll :: String -> Tensor s t -> Tensor '[] t
reduceAll op = unOp ("tf.reduce_" ++ op)

reduceMeanAll :: ∀ (s :: Shape) t. Tensor s t -> Tensor '[] t
reduceMeanAll = reduceAll "mean"

reduce :: ∀ s s' n t. KnownLen s' => String -> Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduce op (T x) = T (funcall ("tf.reduce_" ++ op) [x, text "axis=" <> integer (shapeLen @ s')])

reduceSum, reduceMean :: ∀ s s' n t. KnownLen s' => Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduceSum = reduce @s @s' @n "sum"
reduceMean = reduce @s @s' @n "mean"

reduceSum0 :: ∀ s' n. KnownLen s' => Tensor (n ': s') Float32 -> Tensor s' Float32
reduceSum0 = reduceSum @'[]

add :: ∀ s d t. Tensor (d++s) t -> Tensor d t -> Tensor (d++s) t -- note ++s for for 'broadcasting'
add = binOp "tf.add"

-- add_n :: ∀ s t. [Tensor s t] -> Tensor s t
-- add_n = error "add_n not implemented"

(+) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(+) = add @s @d

(⊕) :: ∀  (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊕) = binOp "tf.add"

(⊝) :: ∀ (d :: Shape) (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊝) = binOp "tf.subtract"

multiply :: Tensor d t -> Tensor d t -> Tensor d t
multiply = binOp "tf.multiply"

equal :: Tensor d t -> Tensor d t -> Tensor d TFBool
equal = binOp "tf.equal"

(⊙) :: ∀ (d :: Shape) t. Tensor d t -> Tensor d t -> Tensor d t
(⊙) = multiply

matmul :: Tensor (o ': n ': s) t -> Tensor (m ': o ': s) t -> Tensor (m ': n ': s) t
matmul = binOp "tf.matmul"

sigmoid, tanh, log, relu :: ∀ s. Tensor s Float32 -> Tensor s Float32
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.log"
relu = unOp "tf.nn.relu"

split0 :: ∀ m n batchShape t. (KnownNat n, KnownNat m, KnownLen batchShape) =>
          Tensor ((n + m) ': batchShape) t -> Gen (Tensor (n ': batchShape) t, Tensor (m ': batchShape) t)
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> text "," <> v2 <> text " = " <> funcall "tf.split" [x, list [showDim @ n, showDim @ m], text "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)

concat0 :: ∀ ys d1 d2 t. (KnownShape ys) =>  T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 t u =
  let T x = t
      T y = u
  in (T (funcall "tf.concat" [list [x,y], text "axis=" <> integer (shapeLen @ ys)]))

expandDim :: ∀ s0 s t. KnownLen s => Tensor (s0 ++ s) t -> Tensor (s0 ++ (1 ': s)) t
expandDim (T x) = (T (funcall "tf.expand_dims" [x, text "axis=" <> integer (shapeLen @ s)]))

expandDim' :: forall n s t. KnownLen s => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim' (T x) = (T (funcall "tf.expand_dims" [x, text "axis=" <> integer (shapeLen @ s)]))

expandDim0 :: ∀ s t. KnownShape s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim' @Dim0

expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim' @Dim1

squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, text "axis=" <> integer (shapeLen @ s1)])

squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]

reshape2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
reshape2 (T t) = T (funcall "tf.reshape" [t, showShape @(m*n ': s)])

unstack :: ∀ s (n::Nat) t. (KnownShape s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack (T x) = do
  v <- newVar
  v <-- T (funcall "tf.unstack" [x, text "axis=" <> integer (shapeLen @ s)])
  return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

stack :: ∀ s (n::Nat) t. (KnownShape s) => V n (T s t) -> Tensor (n ': s) t
stack (V xs) = T (funcall "tf.stack" [(list [x | T x <- xs]), text "axis=" <> integer (shapeLen @ s)])

transpose :: ∀ s t. T (Reverse s) t -> T s t
transpose = unOp "tf.transpose"

gather :: ∀s n indexShape t. T (s ++ '[n]) t -> T indexShape Int32 -> T (s ++ indexShape) t
gather = binOp "tf.gather"

negate :: ∀ s t. T s t -> T s t
negate = unOp "-"

-- convolutionNWC :: forall outputChannels  inputSpatialShape inChannels batchSize t.
--                   T ('[inChannels] ++ inputSpatialShape ++ '[batchSize]) t ->
--                   T ('[outputChannels,inChannels] ++ filterSpatialShape) t ->
--                   T ('[outputChannels] ++ inputSpatialShape ++ '[batchSize]) t
-- convolutionNWC (T input) (T filters) = T (funcall "tf.convolution" [input,filters,named "data_format" (text "NWC")])

convolutionNWC' :: forall outputChannels filterSpatialShape inChannels s t.
                  ((1 + Length filterSpatialShape) ~ Length s) => -- the last dim of s is the batch size
                  T ('[inChannels] ++ s) t ->
                  T ('[outputChannels,inChannels] ++ filterSpatialShape) t ->
                  T ('[outputChannels] ++ s) t
convolutionNWC' (T input) (T filters) = T (funcall "tf.convolution" [input,filters,named "data_format" (text "NWC")])

-- poolNC :: forall dim s inputSpatialShape channels batchSize t.
--                   (inputSpatialShape ~ Take dim s, '[batchSize] ~ Drop dim s) =>
--                   T ('[channels] ++ s) t ->
--                   Vec dim  -> String -> String -> 
--                   T ('[channels] ++ s) t
-- poolNC (T input) windowShape poolingType padding =
--    T (funcall "tf.nn.pool" [input,list (map float (vecToList windowShape)),text poolingType,text padding,named "data_format" (text "NWC")])

-- Difficulty: relate windowSize, inputSpatialShape, outputSpatialShape



softmax0 :: T (n ': s) Float32 -> T (n ': s) Float32
softmax0 = unOp "tf.nn.softmax"

round :: T s ('Typ 'Float w) -> T s ('Typ 'Float w)
round = unOp "tf.round"


argmax0 :: forall n s. KnownLen s => T (n ': s) Float32 -> T s Int64
argmax0 (T t) = T (funcall "tf.argmax" [t, showShapeLen @ s])


cast :: forall u s t. KnownTyp u => KnownLen s => T s t -> T s u
cast (T t) = T (funcall "tf.cast" [t, showTyp @ u])


softmaxCrossEntropyWithLogits :: Tensor '[numClasses,batchSize] Float32 -> Tensor '[numClasses,batchSize] Float32 -> Tensor '[batchSize] Float32
softmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])


oneHot :: forall numClasses w batchSize. KnownNat numClasses =>
          Tensor '[batchSize] ('Typ 'Int w) -> Tensor '[numClasses,batchSize] Float32
oneHot (T indices) = T (funcall "tf.one_hot" [indices, named "depth" (showDim @numClasses), named "axis" (int 0)])

-------------------------
-- Generic parameters

class Parameter p where
  parameter :: String -> Gen p

instance KnownShape shape => Parameter (T shape t) where
  parameter s = parameter' s zeros

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s = (,) <$> parameter (s<>"_fst") <*> parameter (s<>"_snd")

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s = (,,) <$> parameter (s<>"_1") <*> parameter (s<>"_2") <*> parameter (s<>"_3")

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s = (,,,) <$> parameter (s<>"_1") <*> parameter (s<>"_2") <*> parameter (s<>"_3") <*> parameter (s<>"_4")

-- Local Variables:
-- dante-project-root: ".."
-- End:
