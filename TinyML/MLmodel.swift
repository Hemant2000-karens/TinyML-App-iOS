//
//  MLmodel.swift
//  TinyML
//
//  Created by Hemant Kumar on 24/07/24.
//

import Foundation
import TensorFlowLite
import CoreML
import UIKit


class MLmodel {
    private var interpreter: Interpreter

    init(modelName: String) {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "tflite") else {
            fatalError("Failed to load the model file.")
        }

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
        } catch {
            fatalError("Failed to create the interpreter with error: \(error.localizedDescription)")
        }
    }

    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> String {
        guard (try? interpreter.input(at: 0)) != nil else {
            return "Failed to get input tensor"
        }

        let grayscaleData = preprocess(pixelBuffer: pixelBuffer)
        
        do {
            try interpreter.copy(grayscaleData, toInputAt: 0)
            try interpreter.invoke()
        } catch {
            return "Failed to invoke interpreter: \(error.localizedDescription)"
        }
        
        guard let outputTensor = try? interpreter.output(at: 0) else {
            return "Failed to get output tensor"
        }
        
        let results = [Float](unsafeData: outputTensor.data)
        return interpretResults(results: results)
    }

    private func preprocess(pixelBuffer: CVPixelBuffer) -> Data {
        let width = 224
        let height = 224
        var grayscaleData = Data(count: width * height * MemoryLayout<Float32>.size)
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            fatalError("Failed to get base address of pixel buffer")
        }
        
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                let r = Float(buffer[pixelIndex]) / 255.0
                let g = Float(buffer[pixelIndex + 1]) / 255.0
                let b = Float(buffer[pixelIndex + 2]) / 255.0
                let grayscale = (0.299 * r + 0.587 * g + 0.114 * b)
                let grayscaleIndex = y * width + x
                let grayscaleFloat32 = grayscale as Float32
                withUnsafeBytes(of: grayscaleFloat32) { grayscaleData.replaceSubrange(grayscaleIndex * MemoryLayout<Float32>.size..<grayscaleIndex * MemoryLayout<Float32>.size + MemoryLayout<Float32>.size, with: $0) }
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        return grayscaleData
    }

    private func interpretResults(results: [Float]) -> String {
        guard let maxIndex = results.argmax() else {
            return "Failed to interpret results"
        }
        
        let labels = ["Circle","Kite","Parallelogram","Rectangle","Rhombus", "Square","Trapezoid","Rhombus"] // Example labels
        return labels[maxIndex]
    }
}

extension Array where Element: Comparable {
    func argmax() -> Int? {
        guard let maxValue = self.max() else { return nil }
        return self.firstIndex(of: maxValue)
    }
}

extension Array where Element == Float {
    init(unsafeData: Data) {
        self = unsafeData.withUnsafeBytes {
            Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: unsafeData.count / MemoryLayout<Float>.size))
        }
    }
}
