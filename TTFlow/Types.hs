{-# LANGUAGE ConstraintKinds #-}
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
import Data.Char (toLower)
-- import GHC.Prim (unsafeCoerce#)
type DOC = Doc ()

type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

type family Last xs where
  Last '[x] = x
  Last (x ': xs) = Last xs

type family Init xs where
  Init '[x] = '[]
  Init (x ': xs) = x ': Init xs

-- initLast' :: forall s k. ((Init s ++ '[Last s]) ~ s => k) -> k
-- initLast' k = unsafeCoerce# k -- why not?

initLast' :: forall s k. SShape s -> ((Init s ++ '[Last s]) ~ s => k) -> k
initLast' (Cons _ Nil) k = k
initLast' (Cons _ (Cons y ys)) k = initLast' (Cons y ys) (k)

initLast :: forall s k. KnownShape s => ((Init s ++ '[Last s]) ~ s => k) -> k
initLast = initLast' @s shapeSing

type family Length xs where
  Length '[] = 0
  Length (x ': xs) = 1 + Length xs

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

data V (n::Nat) a = V [a]
  deriving (Functor, Foldable, Traversable)

data V' (n::Nat) a where
  VZ :: V' 0 a
  VS :: a -> V' n a -> V' (1+n) a

data Peano = Zero | Succ Peano

type Dim0 = 'Zero
type Dim1 = 'Succ Dim0
type Dim2 = 'Succ Dim1

data SPeano n where
  SZero :: SPeano zero
  SSucc :: SPeano n -> SPeano ('Succ n)

data Vec (n::Peano) a where
  VNil  :: Vec 'Zero a
  VCons :: a -> Vec n a -> Vec ('Succ n) a

vecToList :: Vec n a -> [a]
vecToList VNil = []
vecToList (VCons x xs) = x : vecToList xs

type family App n (xs :: Vec n a) ys where
   App 'Zero 'VNil  xs            =  xs
   App ('Succ n) ('VCons x xs) ys =  x ': App n xs ys

type family Take n xs where
   Take 'Zero xs            =  '[]
   Take ('Succ n) (x ': xs) =  x ': Take n xs

type family Drop n xs where
   Drop 'Zero xs            =  xs
   Drop ('Succ n) (x ': xs) =  Drop n xs

data Kind = Float | Int | Bool deriving Show
data NBits = B32 | B64 | B1 deriving Show
data Typ = Typ Kind NBits

type Float32 = 'Typ 'Float 'B32
type Int32 = 'Typ 'Int 'B32
type Int64 = 'Typ 'Int 'B64
type TFBool = 'Typ 'Bool 'B1

instance Show Typ where
  show (Typ Bool _)= "tf.bool"
  show (Typ k l) = "tf." ++ map toLower (show k) ++ drop 1 (show l)

showTyp :: forall t. KnownTyp t => DOC
showTyp = text (show (typVal @t))

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
class KnownBits t where
  bitsVal :: NBits

instance KnownBits 'B32 where bitsVal = B32
instance KnownBits 'B64 where bitsVal = B64
instance (KnownBits l, KnownKind k) => KnownTyp ('Typ k l) where
  typVal = Typ (kindVal @k) (bitsVal @l)

class KnownKind t where
  kindVal :: Kind

instance KnownKind 'Float where
  kindVal = Float

instance KnownKind 'Int where
  kindVal = Int


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
x <-- T y = gen (x <> text "=" <>  y)

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
