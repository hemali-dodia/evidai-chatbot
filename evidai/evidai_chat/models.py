from django.db import models

# Create your models here.
class ChatSession(models.Model):
    id = models.BigAutoField(primary_key=True)
    token = models.CharField(max_length=255, default='default_token')
    title = models.CharField(max_length=255, default='Assisting New Query')
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()
    show = models.BooleanField(default=True)
    class Meta:
        db_table = 'chat_sessions'
        managed = True

class Conversation(models.Model):
    id = models.BigAutoField(primary_key=True)
    token = models.CharField(max_length=255, default='default_token')
    chat_session_id = models.BigIntegerField()
    question = models.TextField()
    answer = models.TextField()
    prompt_id = models.BigIntegerField(null=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        db_table = 'conversations'
        managed = True 

class BasicPrompts(models.Model):
    id = models.BigAutoField(primary_key=True)
    prompt_category = models.TextField()
    prompt = models.TextField()
    asset_name = models.TextField(null=True)
    asset_sub_cat = models.TextField(null=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()
    
    def __str__(self):
        return self.prompt
    
    class Meta:
        db_table = 'evidai_prompts'
        managed = True

class User(models.Model):
    id = models.AutoField(primary_key=True)
    company_id = models.IntegerField(null=True, blank=True)
    email = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=180)
    email_verified = models.BooleanField(default=False)
    email_verification_token = models.CharField(max_length=255, blank=True, null=True)
    wallet_balance = models.DecimalField(max_digits=16, decimal_places=3, default=0.000)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()
    email_verification_token_expiry = models.DateTimeField(blank=True, null=True)
    last_active_session = models.DateTimeField(blank=True, null=True)
    password_reset_token = models.CharField(max_length=255, blank=True, null=True)
    password_reset_token_expiry = models.DateTimeField(blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    invite_code = models.CharField(max_length=255, unique=True, blank=True, null=True)
    investment_personality_points = models.IntegerField(null=True, blank=True)
    investment_limit = models.CharField(max_length=255, blank=True, null=True)
    referral_code = models.CharField(max_length=6, unique=True, blank=True, null=True)
    referral_id = models.IntegerField(null=True, blank=True)
    introducer_fees = models.FloatField(default=0.0)
    is_introducer = models.BooleanField(default=False)
    two_factor_authentication_type = models.TextField(
        choices=[
            ('SMS', 'SMS'),
            ('EMAIL', 'Email'),
            ('AUTHENTICATOR', 'Authenticator'),
        ],
        blank=True,
        null=True
    )
    two_factor_secret = models.CharField(max_length=100, blank=True, null=True)
    total_referred = models.IntegerField(default=0)
    is_distributor = models.BooleanField(default=False)
    has_password_changed = models.BooleanField(default=True)
    distributor_id = models.IntegerField(null=True, blank=True)
    hkd_wallet_balance = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'users'
        managed = False
        unique_together = ('email', 'invite_code', 'referral_code')
        constraints = [
            models.CheckConstraint(
                check=models.Q(two_factor_authentication_type__in=['SMS', 'EMAIL', 'AUTHENTICATOR']),
                name='users_two_factor_authentication_type_check'
            )
        ]

class UserChatLogin(models.Model):
    id = models.BigAutoField(primary_key=True)
    user_id = models.IntegerField()
    token = models.TextField()

    class Meta:
        db_table = "user_chat_evidai"
        managed = False
