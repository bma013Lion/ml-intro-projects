from titanic_analysis_r2 import train_best_model
import pandas as pd

def get_passenger_details():
    """Prompt user for passenger details and validate input."""
    print("\nTitanic Survival Predictor")
    print("==========================")
    print("Please enter the passenger's details:\n")
    
    # Get and validate passenger class
    while True:
        try:
            pclass = int(input("Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd): ").strip())
            if pclass not in [1, 2, 3]:
                print("Please enter 1, 2, or 3")
                continue
            break
        except ValueError:
            print("Please enter a valid number (1, 2, or 3)")
    
    # Get and validate sex
    while True:
        sex = input("Sex (male/female): ").strip().lower()
        if sex not in ['male', 'female']:
            print("Please enter 'male' or 'female'")
            continue
        sex = 1 if sex == 'male' else 0  # Convert to 1 for male, 0 for female
        break
    
    # Get and validate age
    while True:
        try:
            age = float(input("Age: ").strip())
            if age <= 0 or age > 120:
                print("Please enter a valid age (0-120)")
                continue
            break
        except ValueError:
            print("Please enter a valid number for age")
    
    # Fare ranges by class (based on typical Titanic fares)
    FARE_RANGES = {
        1: (30.0, 512.0),  # 1st class
        2: (12.0, 65.0),   # 2nd cl3ass
        3: (7.0, 35.0)     # 3rd class
    }
    
    # Get and validate fare based on class
    while True:
        try:
            min_fare, max_fare = FARE_RANGES[pclass]
            fare_prompt = f"Fare amount (suggested ${min_fare}-${max_fare} for {pclass}st class): "
            fare = float(input(fare_prompt).strip())
            
            if fare < min_fare or fare > max_fare * 1.5:  # Allow 50% over max
                print(f"Warning: Fare is outside typical range for {pclass}st class (${min_fare}-${max_fare})")
                if not input("Continue with this fare? (y/n): ").lower().startswith('y'):
                    continue
            break
        except ValueError:
            print("Please enter a valid number for fare")
    
    return {
        'Pclass': pclass,
        'Sex': sex,  # Already converted to 0/1
        'Fare': fare
    }

def main():
    """Main function to run the Titanic survival predictor."""
    try:
        # Get passenger information
        passenger_data = get_passenger_details()
        
        # Load prediction model
        print("\nAnalyzing passenger data...")
        model, features = train_best_model()
        
        # Create DataFrame with correct column order
        passenger_df = pd.DataFrame([passenger_data])[features]
        
        # Make prediction
        survival_prob = model.predict_proba(passenger_df)[0][1]
        
        # Display Results
        print("\n=== Prediction Results ===")
        print(f"\nPassenger Details:")
        print("-" * 30)
        print(f"Class: {passenger_data['Pclass']} ({'1st' if passenger_data['Pclass'] == 1 else '2nd' if passenger_data['Pclass'] == 2 else '3rd'} class)")
        print(f"Sex: {'Male' if passenger_data['Sex'] == 1 else 'Female'}")
        print(f"Fare: ${passenger_data['Fare']:.2f}")

        print("\nPrediction:")
        print("-" * 30)
        print(f"Survival Probability: {survival_prob:.1%}")
        print(f"Prediction: {'Survived' if survival_prob >= 0.5 else 'Did Not Survive'}")
        
        # Add some interpretation
        if survival_prob > 0.7:
            print("\nThis passenger had a high chance of survival.")
        elif survival_prob < 0.3:
            print("\nThis passenger had a low chance of survival.")
        else:
            print("\nThe prediction for this passenger is uncertain.")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    while True:
        main()
        
        # Ask user if they want to make another prediction
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            print("\nThank you for using the Titanic Survival Predictor!")
            break