import { Router } from 'express';
import { getDashboardOverview } from '../controllers/dashboard.controller';
import { authenticate } from '../middlewares/auth.middleware';

const router = Router();

router.get('/overview', authenticate, getDashboardOverview);

export default router;

