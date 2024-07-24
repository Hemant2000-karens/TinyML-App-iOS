//
//  Coordinator.swift
//  TinyML
//
//  Created by Hemant Kumar on 24/07/24.
//

import Foundation
import AVFoundation
import CoreImage
class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var parent: CameraView
    var modelHandler: MLmodel

    init(parent: CameraView) {
        self.parent = parent
        self.modelHandler = MLmodel(modelName: "shape_classification_model") // Replace with your actual model name
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Preprocess the pixel buffer
        guard let resizedPixelBuffer = self.resizePixelBuffer(pixelBuffer, width: 224, height: 224) else { return }
        guard let grayscalePixelBuffer = self.convertToGrayscale(resizedPixelBuffer) else { return }

        // Debug: Print pixel buffer information
        print("Preprocessed Pixel Buffer: \(grayscalePixelBuffer)")

        // Run inference
        let result = modelHandler.runModel(onFrame: grayscalePixelBuffer)
        DispatchQueue.main.async {
            self.parent.result = result
        }
    }

    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let transform = CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                                          y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))
        let resizedCIImage = ciImage.transformed(by: transform)

        let ciContext = CIContext(options: nil)

        var resizedPixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &resizedPixelBuffer)

        guard status == kCVReturnSuccess, let outputPixelBuffer = resizedPixelBuffer else {
            print("Error: could not create resized pixel buffer")
            return nil
        }

        ciContext.render(resizedCIImage, to: outputPixelBuffer)
        return outputPixelBuffer
    }

    func convertToGrayscale(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let grayscaleFilter = CIFilter(name: "CIPhotoEffectMono")!
        grayscaleFilter.setValue(ciImage, forKey: kCIInputImageKey)
        guard let grayscaleCIImage = grayscaleFilter.outputImage else {
            print("Error: could not create grayscale image")
            return nil
        }

        let ciContext = CIContext(options: nil)

        var grayscalePixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferWidthKey: CVPixelBufferGetWidth(pixelBuffer),
            kCVPixelBufferHeightKey: CVPixelBufferGetHeight(pixelBuffer),
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        let status = CVPixelBufferCreate(kCFAllocatorDefault, CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer), kCVPixelFormatType_32BGRA, attributes, &grayscalePixelBuffer)

        guard status == kCVReturnSuccess, let outputPixelBuffer = grayscalePixelBuffer else {
            print("Error: could not create grayscale pixel buffer")
            return nil
        }

        ciContext.render(grayscaleCIImage, to: outputPixelBuffer)
        return outputPixelBuffer
    }
}
