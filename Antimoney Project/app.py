from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# Load datasets
ckyc_data = pd.read_csv(r'D:\z\project\project\venv\venv\content\ckyc_dataset.csv')
bank_accounts_data = pd.read_csv(r'D:\z\project\project\venv\venv\content\bank_accounts_dataset.csv')
transactions_data = pd.read_csv(r'D:\z\project\project\venv\venv\content\transactions_dataset.csv')

# Helper functions
def get_ckyc_info(ckyc_number):
    return ckyc_data[ckyc_data['ckyc_number'] == int(ckyc_number)]

def get_customer_accounts(ckyc_number):
    return bank_accounts_data[bank_accounts_data['ckyc_number'] == int(ckyc_number)]

def get_transactions(account_numbers, transaction_type=None):
    if transaction_type:
        return transactions_data[(transactions_data['account_number'].isin(account_numbers)) &
                                (transactions_data['transaction_type'] == transaction_type)]
    return transactions_data[transactions_data['account_number'].isin(account_numbers)]

def detect_money_laundering(account_numbers):
    transactions = get_transactions(account_numbers)
    model = IsolationForest(contamination=0.1)
    transaction_amounts = transactions[['amount']]
    transactions['suspicion'] = model.fit_predict(transaction_amounts)
    return transactions[transactions['suspicion'] == -1]

def get_frequent_transactions(account_numbers, by='amount'):
    transactions = get_transactions(account_numbers)
    if by == 'amount':
        return transactions['amount'].value_counts().head()
    elif by == 'party':
        return transactions['transaction_party'].value_counts().head()
    return transactions.groupby(['transaction_party']).size().sort_values(ascending=False).head()

def export_report(account_numbers, filename):
    transactions = get_transactions(account_numbers)
    transactions.to_csv(filename, index=False)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ckyc_info', methods=['POST'])
def ckyc_info():
    ckyc_number = request.form['ckyc_number']
    customer_info = get_ckyc_info(ckyc_number)
    if customer_info.empty:
        return render_template('error.html', message="CKYC number not found.")
    
    customer_accounts = get_customer_accounts(ckyc_number)
    transactions = get_transactions(customer_accounts['account_number'].tolist())
    return render_template('ckyc_info.html', customer_info=customer_info, accounts=customer_accounts, transactions=transactions.to_dict(orient='records'))

@app.route('/account_menu/<int:account_number>', methods=['GET', 'POST'])
def account_menu(account_number):
    if request.method == 'POST':
        option = request.form['option']
        if option == 'all_transactions':
            transactions = get_transactions([account_number])
            return render_template('transactions.html', transactions=transactions)
        elif option == 'deposit':
            transactions = get_transactions([account_number], transaction_type='deposit')
            return render_template('transactions.html', transactions=transactions)
        elif option == 'withdraw':
            transactions = get_transactions([account_number], transaction_type='withdraw')
            return render_template('transactions.html', transactions=transactions)
        elif option == 'frequent_value':
            frequent_amounts = get_frequent_transactions([account_number], by='amount')
            return render_template('frequent.html', frequent_data=frequent_amounts)
        elif option == 'frequent_party':
            frequent_parties = get_frequent_transactions([account_number], by='party')
            return render_template('frequent.html', frequent_data=frequent_parties)
        elif option == 'money_laundering':
            suspicious_transactions = detect_money_laundering([account_number])
            return render_template('transactions.html', transactions=suspicious_transactions)
        elif option == 'filter_date':
            start_date = request.form['start_date']
            end_date = request.form['end_date']
            transactions = get_transactions([account_number])
            transactions = transactions[(transactions['date'] >= start_date) & (transactions['date'] <= end_date)]
            return render_template('transactions.html', transactions=transactions)
        elif option == 'filter_amount':
            min_amount = float(request.form['min_amount'])
            transactions = get_transactions([account_number])
            transactions = transactions[transactions['amount'] >= min_amount]
            return render_template('transactions.html', transactions=transactions)
        elif option == 'export_report':
            filename = request.form['filename']
            export_report([account_number], filename)
            return render_template('success.html', message=f"Report exported to {filename}.")
    return render_template('account_menu.html', account_number=account_number)

@app.route('/inter_bank_transfers', methods=['POST'])
def inter_bank_transfers():
    ckyc_number = request.form['ckyc_number']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    # Get account numbers for the given CKYC number
    customer_accounts = get_customer_accounts(ckyc_number)
    account_numbers = customer_accounts['account_number'].tolist()
    
    # Filter transactions based on account numbers and dates
    transfers = transactions_data[(transactions_data['account_number'].isin(account_numbers)) &
                                  (transactions_data['date'] >= start_date) &
                                  (transactions_data['date'] <= end_date) &
                                  (transactions_data['transaction_type'] == 'deposit')]
    
    transfer_amounts = transfers.groupby(['account_number', 'transaction_party'])['amount'].sum()
    return render_template('inter_bank_transfers.html', transfers=transfer_amounts)

if __name__ == '__main__':
    app.run(debug=True)
