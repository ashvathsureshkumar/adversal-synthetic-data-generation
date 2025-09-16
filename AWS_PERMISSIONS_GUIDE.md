# AWS Permissions Setup Guide

## For Adversarial Synthetic Data Generator Project

### Step 1: Create IAM Group
1. Group name: `AdversarialDataGenerators`
2. Description: "Permissions for Adversarial-Aware Synthetic Data Generator project"

### Step 2: Attach AWS Managed Policies

#### Required Policies:
```
AmazonS3FullAccess
AmazonSageMakerFullAccess
IAMReadOnlyAccess
```

#### Optional but Recommended:
```
CloudWatchLogsReadOnlyAccess
AmazonEC2ReadOnlyAccess
```

### Step 3: Custom Policy (Optional)
If you want more granular control, create a custom policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::adversal-synthetic-data",
                "arn:aws:s3:::adversal-synthetic-data/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:ListRoles",
                "iam:PassRole"
            ],
            "Resource": "*"
        }
    ]
}
```

### Step 4: Create SageMaker Execution Role
You'll also need a SageMaker execution role. Go to:
- IAM → Roles → Create role
- Service: SageMaker
- Use case: SageMaker - Execution
- Policies: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`
- Role name: `SageMakerExecutionRole-AdversarialData`

### What Each Permission Does:

**AmazonS3FullAccess:**
- Store original datasets
- Save generated synthetic data
- Store model artifacts and checkpoints

**AmazonSageMakerFullAccess:**
- Create and manage training jobs
- Deploy models as endpoints
- Run hyperparameter tuning

**IAMReadOnlyAccess:**
- Read role information for SageMaker
- Verify permissions for service calls

**CloudWatchLogsReadOnlyAccess:**
- Monitor training job logs
- Debug model training issues

### Security Best Practices:
1.  Use groups instead of direct user policies
2.  Principle of least privilege
3.  Regular permission audits
4.  Monitor CloudTrail for API usage
5.  Rotate access keys regularly

### After Setup:
1. Add your user to the `AdversarialDataGenerators` group
2. Generate Access Keys for the user
3. Configure AWS CLI: `aws configure`
4. Test with: `python get_aws_info.py`
