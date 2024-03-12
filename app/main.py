#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
from flask import Flask, jsonify

app = Flask(__name__)

employees_list = [
    {
        '_id': '0skX4zgfbx4QHHuX1tl7',
        'name': 'Abby Bacchus',
        'email': 'abby.b@enso.com',
        'gender': 'male',
        'date_of_birth': '14-05-1989'
    },
    {
        '_id': 'xLylnafOcQufnkGf8r1h',
        'name': 'Dudley Gutierrez',
        'email': 'dudley.g@enso.com',
        'gender': 'male',
        'date_of_birth': '25-09-1975'
    },
    {
        '_id': 'P6sTbOGgtj8RpUtOdLC1',
        'name': 'Pansy Jenning',
        'email': 'pansy.j@enso.com',
        'gender': 'female',
        'date_of_birth': '10-05-1965'
    },
    {
        '_id': 'V1AtNWnAMLgfUml39Niy',
        'name': 'Samuel Castro',
        'email': 'samuel.c@enso.com',
        'gender': 'male',
        'date_of_birth': '18-03-1985'
    },
    {
        '_id': '6vwjMOYBGwMTLfRswYFx',
        'name': 'Toby Morrison',
        'email': 'toby.m@enso.com',
        'gender': 'male',
        'date_of_birth': '28-01-1995'
    }
]


@app.route("/")
def index():
    return jsonify([
        {
            "method": "GET",
            "endpoint": "/api/v1/employees/list",
            "response": "returns list of employees details"
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/employees/<string:employee_id>",
            "response": "returns employee information based on employee id"
        }
    ])


@app.route("/api/v1/employees/list")
def get_employees_list():
    return jsonify(employees_list), 200


@app.route("/api/v1/employees/<string:employee_id>")
def get_employee_by_id(employee_id):
    for employee in employees_list:
        if employee_id in employee['_id']:
            return jsonify(employee), 200
        return jsonify({'error': 'Employee does not exist'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)