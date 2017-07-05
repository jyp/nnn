{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}

import Prelude hiding (tanh)

import GHC.TypeLits
-- import GHC.TypeLits.KnownNat
import Data.Proxy
import Data.Monoid
import Control.Monad (ap)
------------------
-- TYPES 
type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

data V (n::Nat) a = V [a]
  deriving (Functor, Foldable, Traversable)

type Shape = [Nat]
data T (shape :: [Nat]) where
  T :: String -> T shape

data SNat (n :: Nat) where
  SNat :: KnownNat n => Proxy n -> SNat n

data SShape s where
  Nil :: SShape '[]
  Cons :: SNat x -> SShape xs -> SShape (x ': xs)

class KnownShape s where
  shapeSing :: SShape s

instance KnownShape '[] where
  shapeSing = Nil

instance (KnownNat x, KnownShape xs) => KnownShape (x ': xs) where
  shapeSing = Cons (SNat Proxy) shapeSing

getShape :: ∀s. KnownShape s=> SShape s
getShape = shapeSing

shapeToList' :: SShape s -> [Integer]
shapeToList' Nil = []
shapeToList' (Cons (SNat x) xs) = natVal x : shapeToList' xs


shapeToList :: ∀(s::Shape). KnownShape s => [Integer]
shapeToList = shapeToList' (getShape @ s)


showShape :: forall (s :: Shape). KnownShape s => String
showShape = show (shapeToList @ s)

shapeLen :: ∀ (s::Shape). KnownShape s => Int
shapeLen = length (shapeToList @ s)

showShapeLen :: ∀ (s::Shape). KnownShape s => String
showShapeLen = show (shapeLen @ s)

rememberNat :: SNat n -> (KnownNat n => r) -> r
rememberNat (SNat _) k = k


showNat :: ∀(n::Nat). KnownNat n => String
showNat = show (natVal (Proxy @ n))

--------------------------------
-- Generation Effects

type Effect = Integer -> [String]

newtype Gen x = Gen {fromGen ::  (x -> Effect) -> Effect} deriving (Functor)

instance Applicative Gen where
  pure = return
  (<*>) = ap
  
instance Monad Gen where
  return x = Gen $ \k -> k x
  Gen m >>= f = Gen $ \k -> m $ \a -> fromGen (f a) $ \b -> k b

newVar :: Gen String
newVar = Gen $ \ k n -> k ("var" <> show n) (1 + n)

gen :: String -> Gen ()
gen s = Gen $ \ k n -> s : k () n

type Tensor shape = T shape

-----------------------------------------
-- Generation helpers

parens :: String -> String
parens x = "(" <> x <> ")"

brackets :: String -> String
brackets x = "[" <> x <> "]"

commas :: [String] -> String
commas [] = ""
commas xs = foldr (\x y -> x <> ", " <> y) "" xs

list :: [String] -> String
list = brackets . commas

funcall :: String -> [String] -> String
funcall f args = f <> (parens (commas args))

binOp :: forall s1 s2 s3. String -> Tensor s1 -> Tensor s2 -> Tensor s3
binOp op (T x) (T y) = T (funcall op [ x , y])

unOp :: forall s1 s2. String -> Tensor s1 -> Tensor s2
unOp op (T x) = T (funcall op [x])

--------------------------
-- TF primitives

-- | Declare a parameter to optimize.
parameter' :: forall (shape :: [Nat]). KnownShape shape => String -> Gen (T shape)
parameter' name = do -- FIMXE: initialization function
  gen (name <> " = tf.Variable(tf.zeros(" <> (showShape @ shape) <> ")) ") 
  return (T name)

add_n :: ∀ d s. Tensor (d++s) -> Tensor d -> Tensor (d++s) -- note ++s for for 'broadcasting'
add_n = binOp "tf.add_n"

(⊕) :: forall (d :: [Nat]) (s :: [Nat]). Tensor (d ++ s) -> Tensor d -> Tensor (d ++ s)
(⊕) = add_n @d @s

multiply :: Tensor d -> Tensor d -> Tensor d
multiply = binOp "tf.multiply"

(⊙) :: forall (d :: [Nat]). Tensor d -> Tensor d -> Tensor d
(⊙) = multiply

matmul :: Tensor (o ': n ': baTensorchShape) -> Tensor (m ': o ': baTensorchShape) -> Tensor (m ': n ': baTensorchShape)
matmul = binOp "tf.matmul"


sigmoid :: forall s. Tensor s -> Tensor s
sigmoid = unOp "tf.sigmoid"

tanh :: forall s. Tensor s -> Tensor s
tanh = unOp "tf.tanh"

split0 :: forall m n batchShape. (KnownNat n, KnownNat m, KnownShape batchShape) => Tensor ((n + m) ': batchShape) -> Gen (Tensor (n ': batchShape) , Tensor (m ': batchShape))
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> "," <> v2 <> " = " <> funcall "tf.split" [x, list [show n, show m], "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)
  where n = natVal (Proxy @ n)
        m = natVal (Proxy @ m)

concat0 :: forall ys d1 d2. (KnownShape ys) =>  T (d1 ': ys) -> T (d2 ': ys) -> T ((d1 + d2) ': ys)
concat0 t u =
  let T x = t
      T y = u
  in (T (funcall "concat" [brackets (commas [x,y]), "axis=" <> show axis]))
  where axis = shapeLen @ ys -- check

expandDim :: forall s0 batchShape. KnownShape batchShape => Tensor (s0 ++ batchShape) -> Tensor (s0 ++ (1 ': batchShape))
expandDim (T x) = (T (funcall "tf.expand_dims" [x, "axis=" <> show (shapeLen @ batchShape)]))

expandDim0 :: forall batchShape. KnownShape batchShape => Tensor batchShape -> Tensor ((1 ': batchShape))
expandDim0 = expandDim @ '[]


squeeze :: forall s0 s1. KnownShape s1 => Tensor (s0 ++ (1 ': s1)) -> Tensor (s0 ++ s1)
squeeze (T x) = T (funcall "tf.squeeze" [x, "axis=" <> show (shapeLen @ s1)])

squeeze0 :: forall batchShape. KnownShape batchShape => Tensor (1 ': batchShape) -> Tensor batchShape
squeeze0 = squeeze @ '[]


unstack :: forall batchShape (n::Nat). (KnownShape batchShape, KnownNat n) => Tensor (n ': batchShape) -> Gen (V n (T batchShape))
unstack (T x) = do
  v <- newVar
  gen (v <> " = " <> funcall "tf.unstack" [x, "axis=" <> show (shapeLen @ batchShape)] )
  return $ V $ [ T $ v <> brackets (show i)| i <- [0..n-1] ]
        where n = natVal (Proxy @ n)

stack :: forall batchShape (n::Nat). (KnownShape batchShape) => V n (T batchShape) -> Tensor (n ': batchShape) 
stack (V xs) = T (funcall "tf.stack" [(list [x | T x <- xs]), "axis=" <> show (shapeLen @ batchShape)])

transpose :: forall s. T (Reverse s) -> T s
transpose = unOp "tf.transpose"



-------------------------
-- Generic parameters

class Parameter p where
  parameter :: String -> Gen p

instance KnownShape shape => Parameter (T shape) where
  parameter = parameter'

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s = (,) <$> parameter (s<>".fst") <*> parameter (s<>".snd")

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s = (,,) <$> parameter (s<>".1") <*> parameter (s<>".2") <*> parameter (s<>".3")

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s = (,,,) <$> parameter (s<>".1") <*> parameter (s<>".2") <*> parameter (s<>".3") <*> parameter (s<>".4")

--------------------
-- "Contrib"


matvecmulBatch :: forall batchShape cols rows. (KnownNat cols, KnownNat rows, KnownShape batchShape) =>  Tensor (cols ': rows ': batchShape) -> Tensor (cols ': batchShape) -> Tensor (rows ': batchShape)
matvecmulBatch m v = squeeze0 (matmul m (expandDim0 v))

matvecmul :: Tensor (cols ': rows ': '[]) -> Tensor (cols ': batchSize ': '[]) -> Tensor (rows ': batchSize ': '[])
matvecmul m v = matmul v (transpose m)

(∙) :: Tensor '[cols, rows] -> Tensor '[cols,batchSize] -> Tensor '[rows,batchSize] 
(∙) = matvecmul

-- A linear function form a to b is a matrix and a bias.
type a ⊸ b = (Tensor '[a,b], Tensor '[b])

-- | Apply a linear function
(#) :: (a ⊸ b) -> T '[a,batchSize] -> Tensor '[b,batchSize]
(weightMatrix, bias) # v = weightMatrix ∙ v ⊕ bias

type RnnCell state input output = (state , T input) -> Gen (state , T output)

lstm :: forall n x bs. (KnownNat bs) => SNat n ->
        (((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n),
         ((n + x) ⊸ n)) ->
        RnnCell (T '[n,bs], T '[n,bs]) '[x,bs] '[n,bs]
lstm _ (wf,wi,wc,wo) ((ht1 , ct1) , input) = return ((c , h) , h)
  where  hx :: T '[ n + x, bs ]
         hx = concat0 ht1 input
         f = sigmoid (wf # hx)
         i = sigmoid (wi # hx)
         cTilda = tanh (wc # hx)
         o = sigmoid (wo # hx)
         c = (f ⊙ ct1) ⊕ (i ⊙ cTilda)
         h = o ⊕ tanh c

dense :: (n ⊸ m) -> (∀ x. T x -> T x) -> Tensor '[n, batchSize] -> Tensor '[m, batchSize]
dense lf activation t = activation (lf # t)


-- | Stack two RNN cells
stackLayers :: RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0,s1) a c
stackLayers l1 l2 ((s0,s1),x) = do
  (s0',y) <- l1 (s0,x)
  (s1',z) <- l2 (s1,y)
  return ((s0',s1'),z)

-- | @addAttention attn l@ adds the attention function @attn@ to the
-- layer @l@.  Note that @attn@ can depend in particular on a constant
-- external value @h@ which is the complete input to pay attention to.
-- The type parameter @x@ is the size of the portion of @h@ that the
-- layer @l@ will observe.
addAttention :: KnownShape batchShape => (state -> Tensor (x ': batchShape)) -> RnnCell state ((a+x) ': batchShape) (b ': batchShape) -> RnnCell state (a ': batchShape) (b ': batchShape)
addAttention attn l (s,a) = l (s,concat0 a (attn s))

-- | Any pure function (layer) can be transformed into a cell by
-- ignoring the RNN state.
timeDistribute :: (Tensor (a ': batchShape) -> Tensor (b ': batchShape)) -> RnnCell () (a ': batchShape) (b ': batchShape)
timeDistribute pureLayer (s,a) = return (s, pureLayer a)

-- | Build a RNN by repeating a cell @n@ times.
rnn :: forall n state input output.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state input output ->
       (state , Tensor (n ': input)) -> Gen (state , Tensor (n ': output))
rnn cell (s0, t) = do
  xs <- unstack t
  (sFin,us) <- chain cell (s0,xs)
  return (sFin,stack us)

-- | RNN helper
chain :: forall state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chain _ (s0 , V []) = return (s0 , V [])
chain f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (sFin,V xs') <- chain f (s1 , V xs)
  return (sFin,V (x':xs'))

