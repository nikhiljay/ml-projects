//
//  ViewController.swift
//  Boston-App
//
//  Created by Nikhil D'Souza on 3/24/18.
//  Copyright © 2018 Nikhil D'Souza. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UITextFieldDelegate {

    @IBOutlet weak var crimTextField: UITextField!
    @IBOutlet weak var znTextField: UITextField!
    @IBOutlet weak var indusTextField: UITextField!
    @IBOutlet weak var chasTextField: UITextField!
    @IBOutlet weak var noxTextField: UITextField!
    @IBOutlet weak var rmTextField: UITextField!
    @IBOutlet weak var ageTextField: UITextField!
    @IBOutlet weak var disTextField: UITextField!
    @IBOutlet weak var radTextField: UITextField!
    @IBOutlet weak var taxTextField: UITextField!
    @IBOutlet weak var ptratioTextField: UITextField!
    @IBOutlet weak var bTextField: UITextField!
    @IBOutlet weak var lstatTextField: UITextField!
    @IBOutlet weak var predictionLabel: UILabel!
    
    let model = BostonClassifier()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        crimTextField.delegate = self
        znTextField.delegate = self
        indusTextField.delegate = self
        chasTextField.delegate = self
        noxTextField.delegate = self
        rmTextField.delegate = self
        ageTextField.delegate = self
        disTextField.delegate = self
        radTextField.delegate = self
        taxTextField.delegate = self
        ptratioTextField.delegate = self
        bTextField.delegate = self
        lstatTextField.delegate = self
        
        predictionLabel.isHidden = true
    }

    @IBAction func predictButtonTapped(_ sender: Any) {
        let informationArray: [Double] = [
            (crimTextField.text! as NSString).doubleValue,
            (znTextField.text! as NSString).doubleValue,
            (indusTextField.text! as NSString).doubleValue,
            (chasTextField.text! as NSString).doubleValue,
            (noxTextField.text! as NSString).doubleValue,
            (rmTextField.text! as NSString).doubleValue,
            (ageTextField.text! as NSString).doubleValue,
            (disTextField.text! as NSString).doubleValue,
            (radTextField.text! as NSString).doubleValue,
            (taxTextField.text! as NSString).doubleValue,
            (ptratioTextField.text! as NSString).doubleValue,
            (bTextField.text! as NSString).doubleValue,
            (lstatTextField.text! as NSString).doubleValue,
        ]

        guard let mlMultiArray = try? MLMultiArray(shape:[13,], dataType:MLMultiArrayDataType.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        
        for (index, element) in informationArray.enumerated() {
            mlMultiArray[index] = NSNumber(floatLiteral: element)
        }
        
        let input = BostonClassifierInput(input: mlMultiArray)
        
        guard let priceBostonOutput = try? model.prediction(
            input: input
        ) else {
                fatalError("Unexpected runtime error.")
        }
        
        DispatchQueue.main.async {
            self.predictionLabel.text = String(describing: priceBostonOutput.output[0])
            self.predictionLabel.isHidden = false
        }
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.view.endEditing(true)
    }
}

