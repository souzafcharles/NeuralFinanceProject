import { Component } from '@angular/core';

import { User } from 'src/app/models';
import { AccountService } from 'src/app/services';

@Component({ templateUrl: 'home.component.html' })
export class HomeComponent {
    user: User | null;

    constructor(private accountService: AccountService) {
        this.user = this.accountService.userValue;
    }
}