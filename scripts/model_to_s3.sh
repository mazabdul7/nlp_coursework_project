#!/bin/bash

cd /home/ec2-user/nlp_coursework_project

zip -r gpt-epoch-5.zip sample_data/checkpoint-10500
aws s3 cp gpt-epoch-5.zip s3://nlp-project/
