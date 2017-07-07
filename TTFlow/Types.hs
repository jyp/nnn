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
{-# LANGUAGE OverloadedStrings #-}

module TTFlow.Types where

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude
import Text.PrettyPrint.Compact hiding (Last)
import GHC.TypeLits
import Data.Proxy
import Control.Monad (ap)

type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

type family Last xs where
  Last '[x] = x
  Last (x ': xs) = Last xs

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

data V (n::Nat) a = V [a]
  deriving (Functor, Foldable, Traversable)

data Typ = Float32 | Int32

instance Show Typ where
  show Float32 = "tf.float32"
  show Int32 = "tf.int32"

type Shape = [Nat]

data T (shape :: Shape) (t :: Typ) where
  T :: Doc -> T shape t

data SNat (n :: Nat) where
  SNat :: KnownNat n => Proxy n -> SNat n

data SShape s where
  Nil :: SShape '[]
  Cons :: SNat x -> SShape xs -> SShape (x ': xs)

class KnownLen s => KnownShape s where
  shapeSing :: SShape s

instance KnownShape '[] where
  shapeSing = Nil

instance (KnownNat x, KnownShape xs) => KnownShape (x ': xs) where
  shapeSing = Cons (SNat Proxy) shapeSing

class KnownTyp t where
  typVal :: Typ
instance KnownTyp 'Float32 where
  typVal = Float32

instance KnownTyp 'Int32 where
  typVal = Int32

showTyp :: ∀ t. KnownTyp t => Doc
showTyp = text (show (typVal @t))


class KnownLen s where
  shapeLen :: Integer

instance KnownLen '[] where
  shapeLen = 0

instance KnownLen xs => KnownLen (x ': xs) where
  shapeLen = 1 Prelude.+ shapeLen @ xs


getShape :: ∀s. KnownShape s=> SShape s
getShape = shapeSing

shapeToList' :: SShape s -> [Integer]
shapeToList' Nil = []
shapeToList' (Cons (SNat x) xs) = natVal x : shapeToList' xs

shapeToList :: ∀(s::Shape). KnownShape s => [Integer]
shapeToList = shapeToList' (getShape @ s)

showShape :: ∀ (s :: Shape). KnownShape s => Doc
showShape = list (map showDim' (reverse (shapeToList @ s)))

showShapeLen :: ∀ (s::Shape). KnownLen s => Doc
showShapeLen = (text . show) (shapeLen @ s)

rememberNat :: SNat n -> (KnownNat n => r) -> r
rememberNat (SNat _) k = k

showDim' :: Integer -> Doc
showDim' n = text (if n == -1 then "None" else show n)

showDim :: forall n. KnownNat n => Doc
showDim = showDim' (natVal (Proxy @ n))

--------------------------------
-- Generation Effects

type Effect = Integer -> [Doc]

newtype Gen x = Gen {fromGen ::  (x -> Effect) -> Effect} deriving (Functor)

instance Applicative Gen where
  pure = return
  (<*>) = ap

instance Monad Gen where
  return x = Gen $ \k -> k x
  Gen m >>= f = Gen $ \k -> m $ \a -> fromGen (f a) $ \b -> k b

newVar :: Gen Doc
newVar = Gen $ \ k n -> k (text "var" <> integer n) (1 Prelude.+ n)

gen :: Doc -> Gen ()
gen s = Gen $ \ k n -> s : k () n

type Tensor shape = T shape

-----------------------------------------
-- Generation helpers


(<--) :: ∀ (s :: Shape) t. Doc -> T s t -> Gen ()
x <-- T y = gen (x <> text " = " <> y)

funcall :: String -> [Doc] -> Doc
funcall f args = text f <> parens (sep (punctuate comma args))

binOp :: ∀ s1 s2 s3 t1 t2 t3. String -> Tensor s1 t1 -> Tensor s2 t2 -> Tensor s3 t3
binOp op (T x) (T y) = T (funcall op [ x , y])

unOp :: ∀ s1 s2 t1 t2. String -> Tensor s1 t1 -> Tensor s2 t2
unOp op (T x) = T (funcall op [x])

assign :: ∀s t. (T s t) -> Gen (T s t)
assign x = do
  v <- newVar
  v <-- x
  return (T v)

-- Local Variables:
-- dante-project-root: ".."
-- End:
