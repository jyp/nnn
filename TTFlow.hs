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
-- Effects

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

type Tensor shape = Gen (T shape)

-----------------------------------------


parens :: String -> String
parens x = "(" <> x <> ")"

brackets :: String -> String
brackets x = "[" <> x <> "]"

commas :: [String] -> String
commas [] = ""
commas xs = foldr (\x y -> x <> ", " <> y) "" xs

funcall :: String -> [String] -> String
funcall f args = f <> (parens (commas args))

binOp :: forall s1 s2 s3. String -> Tensor s1 -> Tensor s2 -> Tensor s3
binOp op t u = do
  T x <- t
  T y <- u
  return (T (funcall op [ x , y]))

unOp :: forall s1 s2. String -> Tensor s1 -> Tensor s2
unOp op t = do
  T x <- t
  return (T (funcall op [x]))

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


concat0 :: forall ys d1 d2. (KnownShape ys) =>  Tensor (d1 ': ys) -> Tensor (d2 ': ys) -> Tensor ((d1 + d2) ': ys)
concat0 t u = do
  T x <- t
  T y <- u
  return (T (funcall "concat" [brackets (commas [x,y]), "axis=" <> show axis]))
  where ys :: SShape ys
        ys = shapeSing
        axis = length $ shapeToList ys -- check

shapeLen :: SShape s -> Int
shapeLen = length . shapeToList

expandDim0 :: forall batchShape. KnownShape batchShape => Tensor batchShape -> Tensor (1 ': batchShape)
expandDim0 t = do
  T x <- t
  return (T (funcall "expand_dims" [x, "axis=" <> show (shapeLen s)]))
   where s :: SShape batchShape
         s = shapeSing

squeeze0 :: forall batchShape. KnownShape batchShape => Tensor (1 ': batchShape) -> Tensor batchShape
squeeze0 t = do
  T x <- t
  return (T (funcall "expand_dims" [x, "axis=" <> show (shapeLen s)]))
   where s :: SShape batchShape
         s = shapeSing

unstack :: forall batchShape (n::Nat). (KnownShape batchShape, KnownNat n) => Tensor (n ': batchShape) -> Gen (V n (T batchShape))
unstack t = do
  T x <- t
  v <- newVar
  gen (v <> " = " <> funcall "tf.unstack" [x, "axis=" <> show (shapeLen batchShape)] )
  return $ V $ [ T $ v <> brackets (show i)| i <- [0..n-1] ]
  where batchShape :: SShape batchShape
        batchShape = shapeSing
        nProxy :: Proxy n
        nProxy = Proxy
        n :: Integer
        n = natVal nProxy

stack :: forall batchShape (n::Nat). (KnownShape batchShape, KnownNat n) => V n (Tensor batchShape) -> Tensor (n ': batchShape) 
stack t = do
  T x <- t
  v <- newVar
  gen (v <> " = " <> funcall "tf.unstack" [x, "axis=" <> show (shapeLen batchShape)] )
  return $ V $ [ T $ v <> brackets (show i)| i <- [0..n-1] ]
  where batchShape :: SShape batchShape
        batchShape = shapeSing
        nProxy :: Proxy n
        nProxy = Proxy
        n :: Integer
        n = natVal nProxy

--------------------
-- "Contrib"


matvecmul :: forall batchShape cols rows. (KnownNat cols, KnownNat rows, KnownShape batchShape) =>  Tensor (cols ': rows ': batchShape) -> Tensor (cols ': batchShape) -> Tensor (rows ': batchShape)
matvecmul m v = squeeze0 (matmul m (expandDim0 v))

(∙) :: forall batchShape cols rows. (KnownNat cols, KnownNat rows, KnownShape batchShape) =>  Tensor (cols ': rows ': batchShape) -> Tensor (cols ': batchShape) -> Tensor (rows ': batchShape)
(∙) = matvecmul

type a ⊸ b = (Tensor '[a,b], Tensor '[b])

(#) :: (KnownNat a, KnownNat b) => (a ⊸ b) -> Tensor '[a] -> Tensor '[b]
(weightMatrix, bias) # v = weightMatrix ∙ v ⊕ bias


lstm :: forall n x. (KnownNat x, KnownNat n) => SNat n ->
        ((n + x) ⊸ n) -> 
        ((n + x) ⊸ n) ->
        ((n + x) ⊸ n) ->
        ((n + x) ⊸ n) ->
        ((Tensor '[ n ], Tensor '[ n ]) , Tensor '[ x ]) ->
        ((Tensor '[ n ], Tensor '[ n ]) , Tensor '[ n ])
lstm _ wf wi wc wo ((ht1 , ct1) , input) = ((c , h) , h)
  where  hx :: Tensor '[ n + x ]
         hx = concat0 ht1 input
         f = sigmoid (wf # hx)
         i = sigmoid (wi # hx)
         cTilda = tanh (wc # hx)
         o = sigmoid (wo # hx)
         c = (f ⊙ ct1) ⊕ (i ⊙ cTilda)
         h = o ⊕ tanh c

