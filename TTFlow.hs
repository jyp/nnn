{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}


import GHC.TypeLits
-- import GHC.TypeLits.KnownNat
import Data.Proxy

type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

data T (shape :: [Nat]) where
  T :: String -> T shape

data SNat n where
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

rtShape :: SShape s -> [Integer]
rtShape Nil = []
rtShape (Cons (SNat x) xs) = natVal x : rtShape xs

matmul :: T (o ': n ': batchShape) -> T (m ': o ': batchShape) -> T (m ': n ': batchShape)
matmul _ _ = T "matmul"

sigmoid :: forall s. T s -> T s
sigmoid _ = T "sigmoid"

tanh :: forall s. T s -> T s
tanh _ = T "tanh"


concat0 :: forall ys d1 d2. (KnownNat d1, KnownShape ys) =>  T (d1 ': ys) -> T (d2 ': ys) -> T (d1 + d2 ': ys)
concat0 _ _ = T "concat"


expandDim0 :: forall batchShape. KnownShape batchShape => T batchShape -> T (1 ': batchShape)
expandDim0 _ = T "expandDim"
   where s :: SShape batchShape
         s = shapeSing

squeeze0 :: forall batchShape. T (1 ': batchShape) -> T batchShape
squeeze0 _ = T "squeeze0"

matvecmul :: forall batchShape cols rows. (KnownNat cols, KnownShape batchShape) =>  T (cols ': rows ': batchShape) -> T (cols ': batchShape) -> T (rows ': batchShape)
matvecmul m v = squeeze0 (matmul m (expandDim0 v))
