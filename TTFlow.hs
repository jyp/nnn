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

data V (n::Nat) (a :: *) = V [a]
  deriving (Functor, Foldable, Traversable)

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

shapeToList :: SShape s -> [Integer]
shapeToList Nil = []
shapeToList (Cons (SNat x) xs) = natVal x : shapeToList xs

rememberNat :: SNat n -> (KnownNat n => r) -> r
rememberNat (SNat _) k = k

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

add_n :: Tensor d -> Tensor d -> Tensor d
add_n = binOp "tf.add_n"

(⊕) :: forall (d :: [Nat]). Tensor d -> Tensor d -> Tensor d
(⊕) = add_n

multiply :: Tensor d -> Tensor d -> Tensor d
multiply = binOp "tf.multiply"

(⊙) :: forall (d :: [Nat]). Tensor d -> Tensor d -> Tensor d
(⊙) = multiply

matmul :: Tensor (o ': n ': baTensorchShape) -> Tensor (m ': o ': baTensorchShape) -> Tensor (m ': n ': baTensorchShape)
matmul = binOp "matmul"


sigmoid :: forall s. Tensor s -> Tensor s
sigmoid = unOp "sigmoid"

tanh :: forall s. Tensor s -> Tensor s
tanh = unOp "tanh"

split0 :: forall m n batchShape. (KnownNat n, KnownNat m, KnownShape batchShape) => Tensor ((n + m) ': batchShape) -> Gen (Tensor (n ': batchShape) , Tensor (m ': batchShape))
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> "," <> v2 <> " = " <> funcall "tf.split" [x, list [show n, show m], "axis=" <> show (shapeLen s)])
  return (T v1, T v2)
  where s :: SShape batchShape
        s = shapeSing
        nProxy :: Proxy n
        nProxy = Proxy
        n :: Integer
        n = natVal nProxy
        mProxy :: Proxy m
        mProxy = Proxy
        m :: Integer
        m = natVal mProxy

concat0 :: forall ys d1 d2. (KnownShape ys) =>  T (d1 ': ys) -> T (d2 ': ys) -> T ((d1 + d2) ': ys)
concat0 t u =
  let T x = t
      T y = u
  in (T (funcall "concat" [brackets (commas [x,y]), "axis=" <> show axis]))
  where ys :: SShape ys
        ys = shapeSing
        axis = length $ shapeToList ys -- check

shapeLen :: SShape s -> Int
shapeLen = length . shapeToList

expandDim0 :: forall batchShape. KnownShape batchShape => Tensor batchShape -> Tensor (1 ': batchShape)
expandDim0 (T x) = (T (funcall "expand_dims" [x, "axis=" <> show (shapeLen s)]))
   where s :: SShape batchShape
         s = shapeSing

squeeze0 :: forall batchShape. KnownShape batchShape => Tensor (1 ': batchShape) -> Tensor batchShape
squeeze0 (T x) = T (funcall "expand_dims" [x, "axis=" <> show (shapeLen s)])
   where s :: SShape batchShape
         s = shapeSing

unstack :: forall batchShape (n::Nat). (KnownShape batchShape, KnownNat n) => Tensor (n ': batchShape) -> Gen (V n (T batchShape))
unstack (T x) = do
  v <- newVar
  gen (v <> " = " <> funcall "tf.unstack" [x, "axis=" <> show (shapeLen batchShape)] )
  return $ V $ [ T $ v <> brackets (show i)| i <- [0..n-1] ]
  where batchShape :: SShape batchShape
        batchShape = shapeSing
        nProxy :: Proxy n
        nProxy = Proxy
        n :: Integer
        n = natVal nProxy

stack :: forall batchShape (n::Nat). (KnownShape batchShape) => V n (T batchShape) -> Tensor (n ': batchShape) 
stack (V xs) = T (funcall "tf.stack" [(list [x | T x <- xs]), "axis=" <> show (shapeLen batchShape)])
  where batchShape :: SShape batchShape
        batchShape = shapeSing

--------------------
-- "Contrib"


matvecmul :: forall batchShape cols rows. (KnownNat cols, KnownNat rows, KnownShape batchShape) =>  Tensor (cols ': rows ': batchShape) -> Tensor (cols ': batchShape) -> Tensor (rows ': batchShape)
matvecmul m v = squeeze0 (matmul m (expandDim0 v))

(∙) :: forall batchShape cols rows. (KnownNat cols, KnownNat rows, KnownShape batchShape) =>  Tensor (cols ': rows ': batchShape) -> Tensor (cols ': batchShape) -> Tensor (rows ': batchShape)
(∙) = matvecmul

-- A linear function form a to b is a matrix and a bias.
type a ⊸ b = (Tensor '[a,b], Tensor '[b])

-- | Apply a linear function
(#) :: (KnownNat a, KnownNat b) => (a ⊸ b) -> T '[a] -> Tensor '[b]
(weightMatrix, bias) # v = weightMatrix ∙ v ⊕ bias

type RnnCell state input output = (state , T input) -> Gen (state , T output)

lstm :: forall n x. (KnownNat x, KnownNat n) => SNat n ->
        ((n + x) ⊸ n) -> 
        ((n + x) ⊸ n) ->
        ((n + x) ⊸ n) ->
        ((n + x) ⊸ n) ->
        ((T '[ n ], T '[ n ]) , T '[ x ]) ->
        Gen ((T '[ n ], T '[ n ]) , T '[ n ])
lstm _ wf wi wc wo ((ht1 , ct1) , input) = return ((c , h) , h)
  where  hx :: T '[ n + x ]
         hx = concat0 ht1 input
         f = sigmoid (wf # hx)
         i = sigmoid (wi # hx)
         cTilda = tanh (wc # hx)
         o = sigmoid (wo # hx)
         c = (f ⊙ ct1) ⊕ (i ⊙ cTilda)
         h = o ⊕ tanh c

-- dense :: (n ⊸ m) -> (∀ x. T x -> T x) -> Tensor (n ': batchShape) -> Tensor (m ': batchShape)
-- dense lf activation t = activation (lf # t)

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

-- | Any pure function can be transformed into a cell by ignoring the RNN state.
timeDistribute :: (Tensor (a ': batchShape) -> Tensor (b ': batchShape)) -> RnnCell () (a ': batchShape) (b ': batchShape)
timeDistribute pureLayer (s,a) = return (s, pureLayer a)

-- | Build a RNN by repeating a cell @n@ times.
rnn :: forall state input output n.
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

