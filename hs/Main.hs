{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{-
  Simple Neural Network for Handwriting Recognition
  Written in pure Haskell - no ML libraries!

  Architecture: Input(784) -> Hidden(128, ReLU) -> Output(36, Softmax)
  Recognizes: A-Z (0-25) and 0-9 (26-35)

  Training data: EMNIST dataset (prepared by prepare_data.py)
-}

module Main where

type Vector = [] Double
type Matrix = [] [] Double

data NeuralNetwork = NeuralNetwork
  { weightsIH :: !Matrix    -- Input to Hidden (784 x 128)
  , biasH     :: !Vector    -- Hidden bias (129)
  , weightsHO :: !Matrix    -- Hidden to Output (128 * 36)
  , biasO     :: !Vector    -- Output bias (36)
  } deriving (Show, Read)

data Prediction = Prediction
  { predChar        :: Char
  , predIndex       :: Int
  , predConfidence  :: Double
  } deriving (Show)

data TrainingSample = TrainingSample
  { pixels :: !Vector
  , label  :: !Int
  } deriving (Show)


