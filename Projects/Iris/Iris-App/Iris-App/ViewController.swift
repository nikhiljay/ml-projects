//
//  ViewController.swift
//  Iris-App
//
//  Created by Nikhil D'Souza on 3/25/18.
//  Copyright © 2018 Nikhil D'Souza. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UITextFieldDelegate {

    @IBOutlet weak var sepalLengthTextField: UITextField!
    @IBOutlet weak var sepalWidthTextField: UITextField!
    @IBOutlet weak var petalLengthTextField: UITextField!
    @IBOutlet weak var petalWidthTextField: UITextField!
    @IBOutlet weak var predictionLabel: UILabel!
    
    let model = IrisClassifier()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        sepalLengthTextField.delegate = self
        sepalWidthTextField.delegate = self
        petalLengthTextField.delegate = self
        petalWidthTextField.delegate = self
        
        predictionLabel.isHidden = true
    }

    @IBAction func predictButtonTapped(_ sender: Any) {
        let informationArray: [Double] = [
            (sepalLengthTextField.text! as NSString).doubleValue,
            (sepalWidthTextField.text! as NSString).doubleValue,
            (petalLengthTextField.text! as NSString).doubleValue,
            (petalWidthTextField.text! as NSString).doubleValue
        ]
        
        guard let mlMultiArray = try? MLMultiArray(shape:[4,], dataType:MLMultiArrayDataType.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        
        for (index, element) in informationArray.enumerated() {
            mlMultiArray[index] = NSNumber(floatLiteral: element)
        }
        
        let input = IrisClassifierInput(input: mlMultiArray)
        
        guard let irisOutput = try? model.prediction(
            input: input
            ) else {
                fatalError("Unexpected runtime error.")
        }
        
        let output_labels = ["setosa", "versicolor", "virginica"]
        
        DispatchQueue.main.async {
            self.predictionLabel.text = output_labels[Int(truncating: irisOutput.output[0])]
            self.predictionLabel.isHidden = false
        }
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.view.endEditing(true)
    }
}

