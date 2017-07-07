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
{-# LANGUAGE OverloadedStrings #-}

module TTFlow.Types where

import Text.PrettyPrint.Compact hiding (Last)
import GHC.TypeLits
import Data.Proxy
import Control.Monad.State

type DOC = Doc ()

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

data T (shape :: Shape) (t :: Typ) = T {fromTensor :: DOC}

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

showTyp :: ∀ t. KnownTyp t => DOC
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

showShape :: ∀ (s :: Shape). KnownShape s => DOC
showShape = list (map showDim' (reverse (shapeToList @ s)))

showShapeLen :: ∀ (s::Shape). KnownLen s => DOC
showShapeLen = (text . show) (shapeLen @ s)

rememberNat :: SNat n -> (KnownNat n => r) -> r
rememberNat (SNat _) k = k

showDim' :: Integer -> DOC
showDim' n = text (if n == -1 then "None" else show n)

showDim :: forall n. KnownNat n => DOC
showDim = showDim' (natVal (Proxy @ n))

--------------------------------
-- Generation Effects

data GState = GState {nextVar :: Integer,
                      genText :: DOC}
newtype Gen x = Gen {fromGen :: State GState x} deriving (Monad, MonadState GState, Functor, Applicative)

newVar :: Gen DOC
newVar = do
  n <- gets nextVar
  modify $ \GState{..} -> GState {nextVar=nextVar+1,..}
  return (text "var" <> integer n)

gen :: DOC -> Gen ()
gen s = modify $ \GState{..} -> GState {genText=genText $$ s,..}

setGen :: DOC -> Gen ()
setGen d = modify $ \GState{..} -> GState {genText=d,..}

withDOC :: forall a. (DOC -> DOC) -> Gen a -> Gen a
withDOC f g = do
  before <- gets genText
  setGen mempty
  x <- g
  after <- gets genText
  setGen (before $$ f after)
  return x

type Tensor shape = T shape

-----------------------------------------
-- Generation helpers


(<--) :: ∀ (s :: Shape) t. DOC -> T s t -> Gen ()
x <-- T y = gen (hang 2 (x <> text " =")  y)

tuple :: [DOC] -> DOC
tuple = parens . sep . punctuate comma

funcall :: String -> [DOC] -> DOC
funcall f args = text f <> tuple args

binOp :: ∀ s1 s2 s3 t1 t2 t3. String -> Tensor s1 t1 -> Tensor s2 t2 -> Tensor s3 t3
binOp op (T x) (T y) = T (funcall op [ x , y])

unOp :: ∀ s1 s2 t1 t2. String -> Tensor s1 t1 -> Tensor s2 t2
unOp op (T x) = T (funcall op [x])

assign :: ∀s t. (T s t) -> Gen (T s t)
assign x = do
  v <- newVar
  v <-- x
  return (T v)

genFun :: forall b. String -> [DOC] -> Gen b -> Gen b
genFun name args body = do
  gen (text "def " <> text name <> tuple args <> text ":")
  withDOC (\b -> text "  " <> b) body


generate :: Gen () -> String
generate s = renderWith (Options 92 (const id)) (genText (execState (fromGen s) (GState {nextVar = 0, genText = mempty})))

named :: String -> DOC -> DOC
named fname x = text (fname <> "=") <> x
-- Local Variables:
-- dante-project-root: ".."
-- End:
