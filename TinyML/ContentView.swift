//
//  ContentView.swift
//  TinyML
//
//  Created by Hemant Kumar on 24/07/24.
//

import SwiftUI
import AVFoundation


struct ContentView: View {
    @State private var result: String = "No result yet"

    var body: some View {
        ZStack {
            CameraView(result: $result)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                Text(result)
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.white)
                    .font(.headline)
                    .cornerRadius(10)
                    .padding()
            }
        }
    }
}

#Preview {
    ContentView()
}
